"""
Functions for displaying directory structures.
"""

import os
from typing import Optional, List, Any
from fnmatch import fnmatch

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("display_structure")

def display_directory_structure(
    path: str,
    max_depth: Optional[int] = None,
    exclude_patterns: Optional[List[str]] = None,
    sort_by: str = "name"
) -> str:
    """
    Display directory structure in a tree-like format.
    
    Args:
        path: Root directory path
        max_depth: Maximum depth to traverse (None for unlimited)
        exclude_patterns: List of glob patterns to exclude
        sort_by: Sort method ('name', 'size', 'modified')
        
    Returns:
        str: Formatted directory structure
    """
    logger.info(f"Displaying directory structure for: {path}")
    
    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        return ""
    
    exclude_patterns = exclude_patterns or []
    result = []
    
    def should_exclude(item_path: str) -> bool:
        return any(fnmatch(item_path, pattern) for pattern in exclude_patterns)
    
    def get_sort_key(item_path: str) -> Any:
        if sort_by == "size":
            return os.path.getsize(item_path) if os.path.isfile(item_path) else 0
        elif sort_by == "modified":
            return os.path.getmtime(item_path)
        return item_path.lower()
    
    def format_size(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    
    def walk_directory(current_path: str, prefix: str = "", depth: int = 0) -> None:
        if max_depth is not None and depth > max_depth:
            return
        
        try:
            items = os.listdir(current_path)
            items.sort(key=lambda x: get_sort_key(os.path.join(current_path, x)))
            
            for i, item in enumerate(items):
                item_path = os.path.join(current_path, item)
                
                if should_exclude(item_path):
                    continue
                
                is_last = i == len(items) - 1
                current_prefix = prefix + ("└── " if is_last else "├── ")
                next_prefix = prefix + ("    " if is_last else "│   ")
                
                try:
                    if os.path.islink(item_path):
                        target = os.readlink(item_path)
                        result.append(f"{current_prefix}{item} -> {target}")
                    elif os.path.isdir(item_path):
                        result.append(f"{current_prefix}{item}/")
                        walk_directory(item_path, next_prefix, depth + 1)
                    else:
                        size = format_size(os.path.getsize(item_path))
                        result.append(f"{current_prefix}{item} ({size})")
                except OSError as e:
                    result.append(f"{current_prefix}{item} [Error: {str(e)}]")
        except OSError as e:
            logger.error(f"Error accessing directory {current_path}: {e}")
            result.append(f"{prefix}[Error: {str(e)}]")
    
    walk_directory(path)
    return "\n".join(result)