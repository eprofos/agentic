"""
Functions for finding and analyzing import statements in code.
"""

import re
from typing import List, Dict

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("find_imports")

def find_imports(code: str, language: str) -> List[Dict[str, str]]:
    """
    Find import statements in code.
    
    Args:
        code: Code to analyze
        language: Programming language of the code
        
    Returns:
        List of dictionaries with import information
    """
    logger.info(f"Finding imports for language: {language}")
    
    imports = []
    
    if language.lower() == "python":
        # Match import statements in Python
        import_patterns = [
            r"import\s+([\w.]+)(?:\s+as\s+(\w+))?",  # import module [as alias]
            r"from\s+([\w.]+)\s+import\s+(.+)"       # from module import ...
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            
            if pattern == import_patterns[0]:
                # Handle "import module [as alias]"
                for module, alias in matches:
                    imports.append({
                        "type": "import",
                        "module": module,
                        "alias": alias if alias else None
                    })
            else:
                # Handle "from module import ..."
                for module, imports_str in matches:
                    # Split imports by comma and handle aliases
                    for import_item in imports_str.split(","):
                        import_item = import_item.strip()
                        
                        if " as " in import_item:
                            name, alias = import_item.split(" as ")
                            imports.append({
                                "type": "from_import",
                                "module": module,
                                "name": name.strip(),
                                "alias": alias.strip()
                            })
                        else:
                            imports.append({
                                "type": "from_import",
                                "module": module,
                                "name": import_item,
                                "alias": None
                            })
    elif language.lower() in ["javascript", "typescript"]:
        # Match import statements in JavaScript/TypeScript
        import_patterns = [
            r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]",  # import { ... } from 'module'
            r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",        # import name from 'module'
            r"import\s+\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]"  # import * as name from 'module'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            
            if pattern == import_patterns[0]:
                # Handle "import { ... } from 'module'"
                for imports_str, module in matches:
                    # Split imports by comma and handle aliases
                    for import_item in imports_str.split(","):
                        import_item = import_item.strip()
                        
                        if " as " in import_item:
                            name, alias = import_item.split(" as ")
                            imports.append({
                                "type": "named_import",
                                "module": module,
                                "name": name.strip(),
                                "alias": alias.strip()
                            })
                        else:
                            imports.append({
                                "type": "named_import",
                                "module": module,
                                "name": import_item,
                                "alias": None
                            })
            elif pattern == import_patterns[1]:
                # Handle "import name from 'module'"
                for name, module in matches:
                    imports.append({
                        "type": "default_import",
                        "module": module,
                        "name": name,
                        "alias": None
                    })
            else:
                # Handle "import * as name from 'module'"
                for alias, module in matches:
                    imports.append({
                        "type": "namespace_import",
                        "module": module,
                        "name": "*",
                        "alias": alias
                    })
    
    logger.info(f"Found {len(imports)} imports")
    return imports