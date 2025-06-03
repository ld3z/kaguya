import os
import requests
import json
from pathlib import Path
import time # For time.monotonic() in Rich's columns
from datetime import datetime
import re
import sys # For sys.exit
from typing import List, Dict, Tuple, Optional, Any, NamedTuple, Set
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn as RichTimeRemainingColumn, # Alias for internal use
    TimeElapsedColumn as RichTimeElapsedColumn,   # Alias for internal use
    TransferSpeedColumn as RichTransferSpeedColumn,
    FileSizeColumn as RichFileSizeColumn,
    SpinnerColumn,
    TaskID,
    Task,
    ProgressColumn # Base class for custom column
)
from rich.live import Live
from rich.panel import Panel
from rich.text import Text as RichText

# --- Constants ---
API_KEY_FILE = Path("api_key.txt")
UPLOAD_RECORD_FILE = "imgchest_upload_record.txt"
MANGA_INFO_FILE = "info.txt"
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
IMGCHEST_API_BASE_URL = "https://api.imgchest.com/v1"
MAX_IMAGES_PER_BATCH = 20

# --- Rich Console ---
console = Console()

# --- NamedTuples for structured data ---
class ChapterInfo(NamedTuple):
    volume: str
    chapter: str
    title: str

class FolderDetails(NamedTuple):
    path: Path
    name: str
    image_count: int

# --- Custom Progress Columns ---
class ConditionalFileSizeColumn(RichFileSizeColumn):
    """Renders file size, but only if task.fields['is_byte_task'] is True."""
    def render(self, task: "Task") -> RichText:
        if task.fields.get("is_byte_task"):
            return super().render(task)
        return RichText("")

class ConditionalTransferSpeedColumn(RichTransferSpeedColumn):
    """Renders transfer speed, but only if task.fields['is_byte_task'] is True."""
    def render(self, task: "Task") -> RichText:
        if task.fields.get("is_byte_task"):
            return super().render(task)
        return RichText("")

class CustomTimeDisplayColumn(ProgressColumn):
    """
    Displays 'Time Elapsed' for item-based tasks (is_byte_task=False)
    and 'Time Remaining' (or total time if finished) for byte-based tasks (is_byte_task=True).
    """
    def __init__(self):
        super().__init__()
        self._time_remaining_col = RichTimeRemainingColumn()
        self._time_elapsed_col = RichTimeElapsedColumn()

    def render(self, task: "Task") -> RichText:
        if task.fields.get("is_byte_task"):  # For byte tasks (e.g., individual batch uploads)
            if task.finished:
                # Show total time taken for the completed byte task
                return self._time_elapsed_col.render(task)
            else:
                # Show ETA for the running byte task
                return self._time_remaining_col.render(task)
        else:  # For item-based tasks (e.g., Overall Progress, Chapter Batches)
            # Always show time elapsed (counts up)
            return self._time_elapsed_col.render(task)


# --- Core Functions (Ensure these are complete from previous steps) ---
def load_api_key(file_path: Path = API_KEY_FILE) -> Optional[str]:
    """Load API key from a text file."""
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        console.print(f"[red]Error: {file_path} not found. Please create it with your API key.[/red]")
        return None
    except IOError as e:
        console.print(f"[red]Error reading API key from {file_path}: {e}[/red]")
        return None

def parse_folder_name(folder_name: str) -> ChapterInfo:
    """
    Parse folder name to extract volume, chapter, and title.
    Attempts multiple regex patterns to cover common naming conventions.
    """
    # Pattern for "V<volume> Ch<chapter> <title>"
    volume_pattern = r'V(\d+)\s+Ch(\d+(?:\.\d+)?)\s*(.*)?'
    volume_match = re.match(volume_pattern, folder_name, re.IGNORECASE)
    if volume_match:
        volume = volume_match.group(1)
        chapter_num = volume_match.group(2)
        title = volume_match.group(3).strip() if volume_match.group(3) else ""
        return ChapterInfo(volume, chapter_num, title)

    # Pattern for "Ch<chapter> <title>" or "Chapter <chapter> <title>"
    chapter_pattern = r'Ch(?:apter)?\s*(\d+(?:\.\d+)?)\s*(.*)?'
    chapter_match = re.match(chapter_pattern, folder_name, re.IGNORECASE)
    if chapter_match:
        chapter_num = chapter_match.group(1)
        title = chapter_match.group(2).strip() if chapter_match.group(2) else ""
        return ChapterInfo("", chapter_num, title)

    # Fallback: extract numbers from the folder name
    numbers = re.findall(r'\d+(?:\.\d+)?', folder_name)
    if len(numbers) >= 2: # Assume first is volume, second is chapter
        return ChapterInfo(numbers[0], numbers[1], "")
    elif len(numbers) == 1: # Assume it's the chapter number
        return ChapterInfo("", numbers[0], "")
    else:
        console.print(f"[yellow]Warning: Could not parse volume/chapter from '{folder_name}'. Defaulting to Ch 1, title='{folder_name}'.[/yellow]")
        return ChapterInfo("", "1", folder_name)

def load_manga_info(base_folder_path: Path) -> Dict[str, str]:
    info_file = base_folder_path / MANGA_INFO_FILE
    info: Dict[str, str] = {
        'title': '', 'description': '', 'artist': '',
        'author': '', 'cover': '', 'groups': ''
    }
    if info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key in info:
                            info[key] = value
        except IOError as e:
            console.print(f"[yellow]Warning: Could not read {MANGA_INFO_FILE}: {e}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: An unexpected error occurred while reading {MANGA_INFO_FILE}: {e}[/yellow]")
    else:
        console.print(f"[yellow]{MANGA_INFO_FILE} not found in {base_folder_path}. Manga metadata will be minimal.[/yellow]")
    return info

def load_upload_record(base_folder_path: Path) -> Dict[str, Dict[str, str]]:
    record_file = base_folder_path / UPLOAD_RECORD_FILE
    uploaded_folders: Dict[str, Dict[str, str]] = {}
    if record_file.exists():
        try:
            with open(record_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '|' in line:
                        parts = line.split('|', 4)
                        if len(parts) >= 4:
                            folder_name, album_url, timestamp, image_count_str = parts[:4]
                            post_id = parts[4] if len(parts) > 4 else album_url.split('/')[-1]
                            uploaded_folders[folder_name] = {
                                'album_url': album_url,
                                'timestamp': timestamp,
                                'image_count': image_count_str,
                                'post_id': post_id
                            }
                        else:
                            console.print(f"[yellow]Warning: Skipping malformed line in {UPLOAD_RECORD_FILE}: {line}[/yellow]")
        except IOError as e:
            console.print(f"[yellow]Warning: Could not read upload record {record_file}: {e}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: An unexpected error occurred while reading {record_file}: {e}[/yellow]")
    return uploaded_folders

def save_upload_record(base_folder_path: Path, uploaded_folders: Dict[str, Dict[str, str]], live: Optional[Live] = None):
    record_file = base_folder_path / UPLOAD_RECORD_FILE
    output_func = live.console.print if live else console.print
    try:
        with open(record_file, 'w', encoding='utf-8') as f:
            f.write("# ImageChest Upload Record\n")
            f.write("# Format: folder_name|album_url|timestamp|image_count|post_id\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for folder_name, data in uploaded_folders.items():
                f.write(f"{folder_name}|{data['album_url']}|{data['timestamp']}|"
                        f"{data.get('image_count', 'unknown')}|{data.get('post_id', data['album_url'].split('/')[-1])}\n")
        output_func(f"[green]Upload record saved to: {record_file}[/green]")
    except IOError as e:
        output_func(f"[red]Error: Could not save upload record to {record_file}: {e}[/red]")

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '_', name)
    return name

def load_manga_json(base_folder_path: Path, manga_title: str, live: Optional[Live] = None) -> Tuple[Dict[str, Any], Path]:
    output_func = live.console.print if live else console.print
    sanitized_title = sanitize_filename(manga_title) if manga_title else "untitled_manga"
    json_file = base_folder_path / f"{sanitized_title}.json"
    manga_json_data: Dict[str, Any]

    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                manga_json_data = json.load(f)
            output_func(f"[dim]Loaded existing manga data from {json_file}[/dim]")
            return manga_json_data, json_file
        except (json.JSONDecodeError, IOError) as e:
            output_func(f"[yellow]Warning: Could not read existing JSON {json_file}: {e}. Creating a new one.[/yellow]")

    manga_json_data = {
        "title": manga_title, "description": "", "artist": "",
        "author": "", "cover": "", "chapters": {}
    }
    return manga_json_data, json_file

def save_manga_json(json_file_path: Path, manga_json_data: Dict[str, Any], live: Optional[Live] = None):
    output_func = live.console.print if live else console.print
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(manga_json_data, f, indent=2, ensure_ascii=False)
        output_func(f"[green]Manga JSON saved to: {json_file_path}[/green]")
    except IOError as e:
        output_func(f"[red]Error: Could not save manga JSON to {json_file_path}: {e}[/red]")

def get_image_files(folder_path: Path, live: Optional[Live] = None) -> List[Path]:
    output_func = live.console.print if live else console.print
    if not folder_path.is_dir():
        output_func(f"[red]Error: Folder '{folder_path}' does not exist or is not a directory.[/red]")
        return []
    image_files = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_files, key=lambda x: x.name.lower())

def find_subfolders_with_images(base_path: Path) -> List[FolderDetails]:
    if not base_path.is_dir():
        console.print(f"[red]Error: Base path '{base_path}' does not exist or is not a directory.[/red]")
        return []
    subfolders: List[FolderDetails] = []
    for item in base_path.iterdir():
        if item.is_dir():
            image_count = sum(1 for f in item.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            if image_count > 0:
                subfolders.append(FolderDetails(item, item.name, image_count))
    return sorted(subfolders, key=lambda x: x.name.lower())

def _perform_imgchest_image_upload(
    url: str, api_key: str, image_files_batch: List[Path],
    progress: Progress, task_description: str
) -> Dict[str, Any]:
    files_to_upload_fields: List[Tuple[str, Tuple[str, Any, str]]] = []
    opened_files: List[Any] = []
    upload_task_id: Optional[TaskID] = None
    try:
        for file_path in image_files_batch:
            try:
                file_handle = open(file_path, 'rb')
                opened_files.append(file_handle)
                files_to_upload_fields.append(('images[]', (file_path.name, file_handle, f'image/{file_path.suffix[1:]}')))
            except IOError as e:
                if upload_task_id is not None: progress.remove_task(upload_task_id)
                return {'success': False, 'error': f"Error opening file {file_path.name}: {e}"}
        if not files_to_upload_fields:
             return {'success': False, 'error': "No valid image files to upload in this batch."}

        encoder = MultipartEncoder(fields=files_to_upload_fields)
        upload_task_id = progress.add_task(
            task_description, total=encoder.len, fields={"is_byte_task": True}
        )
        def progress_callback(monitor: MultipartEncoderMonitor):
            if upload_task_id is not None:
                 progress.update(upload_task_id, completed=monitor.bytes_read)
        monitor = MultipartEncoderMonitor(encoder, progress_callback)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": monitor.content_type}
        response = requests.post(url, data=monitor, headers=headers, timeout=300)
        if response.status_code == 200:
            try:
                data = response.json()
                if 'error' in data or ('status' in data and data['status'] == 'error'):
                    return {'success': False, 'error': data.get('error', data.get('message', 'Unknown API error'))}
                return {'success': True, 'data': data}
            except json.JSONDecodeError:
                return {'success': False, 'error': "Invalid JSON response from API."}
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f"Request failed: {e}"}
    except Exception as e:
        return {'success': False, 'error': f"Unexpected error during upload: {e}"}
    finally:
        if upload_task_id is not None:
            task_index = -1
            for idx, t_id in enumerate(progress.task_ids):
                if t_id == upload_task_id: task_index = idx; break
            if task_index != -1 and not progress.tasks[task_index].finished:
                 progress.update(upload_task_id, completed=progress.tasks[task_index].total)
            progress.remove_task(upload_task_id)
        for fh in opened_files: fh.close()

def upload_initial_batch(image_files_batch: List[Path], api_key: str, chapter_name: str, batch_idx_info: str, progress: Progress) -> Dict[str, Any]:
    url = f"{IMGCHEST_API_BASE_URL}/post"
    task_description = f"[cyan]Upload Batch[/cyan]: {chapter_name} ({batch_idx_info})"
    result = _perform_imgchest_image_upload(url, api_key, image_files_batch, progress, task_description)
    if result['success'] and 'data' in result:
        api_data = result['data'].get('data', {})
        if 'id' in api_data:
            return {'success': True, 'album_url': f"https://imgchest.com/p/{api_data['id']}",
                    'post_id': api_data['id'], 'total_images': len(api_data.get('images', []))}
        else: return {'success': False, 'error': "API response missing post ID."}
    return result

def add_images_to_existing_album(image_files_batch: List[Path], post_id: str, api_key: str, chapter_name: str, batch_idx_info: str, progress: Progress) -> Dict[str, Any]:
    url = f"{IMGCHEST_API_BASE_URL}/post/{post_id}/add"
    task_description = f"[cyan]Upload Batch[/cyan]: {chapter_name} ({batch_idx_info})"
    result = _perform_imgchest_image_upload(url, api_key, image_files_batch, progress, task_description)
    if result['success']:
        return {'success': True, 'added_images': len(image_files_batch)}
    return result

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def upload_all_images_for_chapter(
    image_files: List[Path], api_key: str, chapter_name_for_desc: str,
    progress: Progress, live: Live
) -> Dict[str, Any]:
    if not image_files:
        live.console.print(f"[dim]Info: No image files for '{chapter_name_for_desc}'.[/dim]")
        return {'success': False, 'error': "No image files for upload.", 'total_uploaded': 0}
    image_chunks = chunk_list(image_files, MAX_IMAGES_PER_BATCH)
    total_chunks, total_uploaded_count = len(image_chunks), 0
    post_id: Optional[str] = None; album_url: Optional[str] = None
    chapter_batch_task_id: Optional[TaskID] = None
    try:
        chapter_batch_task_id = progress.add_task(
            f"[blue]Chapter Batches '{chapter_name_for_desc}'[/blue]",
            total=total_chunks, fields={"is_byte_task": False}
        )
        for i, chunk in enumerate(image_chunks):
            batch_info_str = f"Batch {i+1}/{total_chunks}"
            current_op_desc = "Create Album" if i == 0 else "Add Images"
            if chapter_batch_task_id:
                progress.update(chapter_batch_task_id, description=f"[blue]Chapter '{chapter_name_for_desc}'[/blue] ({batch_info_str} - {current_op_desc})")
            if i == 0:
                res = upload_initial_batch(chunk, api_key, chapter_name_for_desc, batch_info_str, progress)
                if not res['success']:
                    live.console.print(f"[red]❌ Error creating album for '{chapter_name_for_desc}': {res.get('error', 'Unknown')}[/red]")
                    return {'success': False, 'error': f"Failed to create album: {res.get('error', 'Unknown')}", 'total_uploaded': 0}
                post_id, album_url = res['post_id'], res['album_url']
                total_uploaded_count += res['total_images']
                live.console.print(f"[green]✓ Album created for '{chapter_name_for_desc}': {album_url} ({res['total_images']} images).[/green]")
            else:
                if not post_id:
                    live.console.print(f"[red]❌ Critical: post_id missing for '{chapter_name_for_desc}'.[/red]")
                    return {'success': False, 'error': "post_id missing", 'total_uploaded': total_uploaded_count}
                time.sleep(1)
                res = add_images_to_existing_album(chunk, post_id, api_key, chapter_name_for_desc, batch_info_str, progress)
                if res['success']:
                    total_uploaded_count += res['added_images']
                    live.console.print(f"[green]✓ Added {res['added_images']} images to '{chapter_name_for_desc}'.[/green]")
                else:
                    live.console.print(f"[red]❌ Error adding batch {i+1} to '{chapter_name_for_desc}': {res.get('error', 'Unknown')}[/red]")
                    return {'success': False, 'error': f"Failed batch {i+1}: {res.get('error', 'Unknown')}",
                            'total_uploaded': total_uploaded_count, 'album_url': album_url, 'post_id': post_id}
            if chapter_batch_task_id: progress.update(chapter_batch_task_id, advance=1)
    finally:
        if chapter_batch_task_id:
            task_index = -1
            for idx, t_id in enumerate(progress.task_ids):
                if t_id == chapter_batch_task_id: task_index = idx; break
            if task_index != -1 and not progress.tasks[task_index].finished:
                 progress.update(chapter_batch_task_id, completed=progress.tasks[task_index].total)
            progress.remove_task(chapter_batch_task_id)
    return {'success': True, 'album_url': album_url, 'post_id': post_id, 'total_uploaded': total_uploaded_count}

def process_single_folder(
    folder_details: FolderDetails, base_folder_path: Path, manga_overall_info: Dict[str, str],
    uploaded_folders_record: Dict[str, Dict[str, str]], api_key: str, progress: Progress, live: Live
) -> bool:
    live.console.print(Panel(RichText(f"Processing: {folder_details.name}", justify="center"), style="bold yellow", border_style="yellow"))
    image_files = get_image_files(folder_details.path, live)
    if not image_files:
        live.console.print(f"[yellow]Warning: No images in {folder_details.name}. Skipping.[/yellow]")
        return False
    chapter_info = parse_folder_name(folder_details.name)
    if folder_details.name in uploaded_folders_record:
        existing = uploaded_folders_record[folder_details.name]
        live.stop()
        console.print(f"\n[yellow]⚠️ '{folder_details.name}' already uploaded![/yellow]")
        console.print(f"   URL: {existing['album_url']}, Date: {existing['timestamp']}, Images: {existing.get('image_count', 'N/A')}")
        skip = console.input(f"\n[bold yellow]Skip ({folder_details.name})? (Y/n):[/bold yellow] ").strip().lower()
        live.start(refresh=True)
        if skip != 'n':
            live.console.print(f"[dim]Skipped '{folder_details.name}'.[/dim]")
            return False
    live.console.print(f"\n[bold]📂 Chapter Info:[/bold] V: {chapter_info.volume or 'N/A'}, Ch: {chapter_info.chapter}, Title: {chapter_info.title or 'N/A'}")
    live.console.print(f"[bold]📸 Found {len(image_files)} image(s).[/bold]", highlight= False)
    upload_res = upload_all_images_for_chapter(image_files, api_key, folder_details.name, progress, live)
    if upload_res['success']:
        live.console.print(f"\n[bold green]🎉 SUCCESS! {upload_res['total_uploaded']} images for '{folder_details.name}'.[/bold green]")
        live.console.print(f"Album URL: {upload_res['album_url']}")
        uploaded_folders_record[folder_details.name] = {
            'album_url': upload_res['album_url'], 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_count': str(upload_res['total_uploaded']), 'post_id': upload_res['post_id']
        }
        save_upload_record(base_folder_path, uploaded_folders_record, live)
        if manga_overall_info.get('title'):
            m_json, json_path = load_manga_json(base_folder_path, manga_overall_info['title'], live)
            for k in ['title', 'description', 'artist', 'author', 'cover']:
                if k in manga_overall_info and manga_overall_info[k]: m_json[k] = manga_overall_info[k]
            ch_data: Dict[str, Any] = {
                "title": chapter_info.title, "last_updated": str(int(time.time())),
                "groups": {manga_overall_info.get('groups', 'UnknownGroup'): f"/proxy/api/imgchest/chapter/{upload_res['post_id']}"}
            }
            if chapter_info.volume: ch_data["volume"] = chapter_info.volume
            if 'chapters' not in m_json: m_json['chapters'] = {}
            m_json['chapters'][chapter_info.chapter] = ch_data
            save_manga_json(json_path, m_json, live)
        return True
    else:
        live.console.print(f"\n[bold red]❌ UPLOAD FAILED for '{folder_details.name}': {upload_res.get('error', 'Unknown')}[/bold red]")
        if upload_res.get('total_uploaded', 0) > 0:
            live.console.print(f"[yellow]Partial success: {upload_res['total_uploaded']} images uploaded.[/yellow]")
            if 'album_url' in upload_res and upload_res['album_url']:
                live.console.print(f"[yellow]Partial Album URL: {upload_res['album_url']}[/yellow]")
        return False

def parse_folder_selection(selection_str: str, num_folders: int) -> Optional[List[int]]:
    selected_indices: Set[int] = set()
    if not selection_str: console.print("[red]Error: No selection.[/red]"); return None
    try:
        for part in selection_str.split(','):
            part = part.strip()
            if '-' in part:
                s, e = map(int, part.split('-', 1))
                if not (1 <= s <= e <= num_folders):
                    console.print(f"[red]Invalid range: {s}-{e}. Max: {num_folders}.[/red]"); return None
                selected_indices.update(range(s - 1, e))
            else:
                idx = int(part)
                if not (1 <= idx <= num_folders):
                    console.print(f"[red]Invalid #: {idx}. Max: {num_folders}.[/red]"); return None
                selected_indices.add(idx - 1)
        return sorted(list(selected_indices))
    except ValueError:
        console.print("[red]Invalid format. Use 1,3,5-7.[/red]"); return None

def main():
    api_key = load_api_key()
    if not api_key: sys.exit(1)
    console.print("[green]API Key loaded.[/green]")

    while True:
        base_path_str = console.input("[bold cyan]Enter the base folder path (containing chapter subfolders, e.g., /path/to/manga):[/bold cyan] ").strip()
        base_folder_path = Path(base_path_str)
        if base_folder_path.is_dir(): break
        console.print(f"[red]Error: '{base_path_str}' not valid dir.[/red]")

    subfolders = find_subfolders_with_images(base_folder_path)
    if not subfolders: console.print("[yellow]No subfolders with images. Exiting.[/yellow]"); sys.exit(0)

    manga_info = load_manga_info(base_folder_path)
    uploaded_record = load_upload_record(base_folder_path)

    console.print("\n[bold underline]📖 Manga Info:[/bold underline]")
    if manga_info.get('title'):
        for k, v in manga_info.items():
            if v: console.print(f"   [dim]{k.capitalize()}:[/dim] {v}")
    else: console.print("   [dim]No info.txt or title missing. Minimal metadata.[/dim]")

    console.print(f"\n[bold underline]📁 Found {len(subfolders)} folder(s):[/bold underline]", highlight= False)
    for i, fd in enumerate(subfolders, 1):
        status = "✓ Uploaded" if fd.name in uploaded_record else "○ New"
        color = "green" if fd.name in uploaded_record else "yellow"
        console.print(f"{i:3d}. {fd.name} ({fd.image_count} images) [[{color}]{status}[/{color}]]")

    console.print("\n[bold underline]⬆️ Upload Options:[/bold underline]\n1. Upload all folders\n2. Upload only new folders (skip already uploaded)\n3. Select specific folder(s) to upload/re-upload\n4. Cancel", highlight=False)
    to_process: List[FolderDetails] = []
    while True:
        choice = console.input("\n[bold cyan]Choose (1-4):[/bold cyan] ").strip()
        if choice == '1': to_process = subfolders; console.print("[green]Selected: All folders.[/green]"); break
        elif choice == '2':
            to_process = [f for f in subfolders if f.name not in uploaded_record]
            if not to_process: console.print("[yellow]No new folders. Exiting.[/yellow]"); sys.exit(0)
            console.print(f"[green]Selected: {len(to_process)} NEW folder(s).[/green]"); break
        elif choice == '3':
            sel_str = console.input("[cyan]Enter folder #s (e.g. 1,3,5-7):[/cyan] ").strip()
            indices = parse_folder_selection(sel_str, len(subfolders))
            if indices is not None:
                to_process = [subfolders[i] for i in indices]
                if not to_process: console.print("[yellow]No valid folders selected.[/yellow]")
                else: console.print(f"[green]Selected: {len(to_process)} specific folder(s).[/green]"); break
        elif choice == '4': console.print("[yellow]Canceled. Exiting.[/yellow]"); sys.exit(0)
        else: console.print("[yellow]Invalid choice (1-4).[/yellow]")

    if not to_process: console.print("[yellow]No folders to process. Exiting.[/yellow]"); sys.exit(0)
    console.print("\n[bold underline]Will process:[/bold underline]")
    for fd in to_process: console.print(f"  - {fd.name}")
    if console.input("\n[bold yellow]Proceed? (y/N):[/bold yellow] ").strip().lower() != 'y':
        console.print("[yellow]Canceled. Exiting.[/yellow]"); sys.exit(0)

    success_ct = 0
    console.print("\n[bold underline]🚀 Starting upload...[/bold underline]")

    # --- Progress Bar Configuration ---
    progress_columns = [
        SpinnerColumn(finished_text="[green]✓[/green]"),
        TextColumn("[progress.description]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("• {task.completed} of {task.total} •"),
        ConditionalTransferSpeedColumn(),
        ConditionalFileSizeColumn(),
        CustomTimeDisplayColumn(), # Use the new custom time column
    ]

    progress_bar_manager = Progress(*progress_columns, console=console, transient=False, expand=True)

    with Live(progress_bar_manager, console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        overall_task_id = progress_bar_manager.add_task(
            "[bold #AAAAFF]Overall Progress[/bold #AAAAFF]",
            total=len(to_process),
            fields={"is_byte_task": False} # Item-based
        )
        for folder_item in to_process:
            if process_single_folder(
                folder_item, base_folder_path, manga_info,
                uploaded_record, api_key, progress_bar_manager, live
            ): success_ct += 1
            progress_bar_manager.update(overall_task_id, advance=1)

        progress_bar_manager.update(overall_task_id, completed=len(to_process))
        progress_bar_manager.update(overall_task_id, description="[bold green]Overall Progress Complete[/bold green]")

    console.print(f"\n[bold green]🎉 Batch complete! {success_ct}/{len(to_process)} processed.[/bold green]")

if __name__ == "__main__":
    main()