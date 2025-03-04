"""
Code-related tools for the Pydantic AI agent system.
These tools help with code generation, analysis, and manipulation.
"""

import os
import re
import ast
import json
from typing import List, Dict, Any, Optional, Union, Tuple

from utils.logging.logger import setup_logger
from tools.utility_tools.common_tools import (
    read_file_content,
    write_file_content,
    detect_language_from_file
)

# Create a logger for this module
logger = setup_logger("code_tools")

def extract_function_definitions(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Extract function definitions from code.
    
    Args:
        code: Code to extract function definitions from
        language: Programming language of the code
        
    Returns:
        List of dictionaries with function information
    """
    logger.info(f"Extracting function definitions for language: {language}")
    
    functions = []
    
    if language.lower() == "python":
        try:
            # Parse the Python code
            tree = ast.parse(code)
            
            # Extract function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function name
                    name = node.name
                    
                    # Get function arguments
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    
                    # Get function docstring if available
                    docstring = ast.get_docstring(node)
                    
                    # Get function body
                    body_lines = []
                    for line in code.splitlines()[node.lineno:node.end_lineno]:
                        body_lines.append(line)
                    body = "\n".join(body_lines)
                    
                    functions.append({
                        "name": name,
                        "args": args,
                        "docstring": docstring,
                        "body": body,
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno
                    })
        except SyntaxError as e:
            logger.error(f"Syntax error in Python code: {e}")
        except Exception as e:
            logger.error(f"Error extracting function definitions from Python code: {e}")
    elif language.lower() == "javascript":
        # Simple regex-based extraction for JavaScript functions
        # This is a simplified approach and might not handle all cases correctly
        
        # Match function declarations: function name(...) {...}
        function_pattern = r"function\s+(\w+)\s*\((.*?)\)\s*\{([\s\S]*?)\}"
        matches = re.findall(function_pattern, code)
        
        for name, args_str, body in matches:
            args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
            functions.append({
                "name": name,
                "args": args,
                "body": body,
                "type": "function_declaration"
            })
        
        # Match arrow functions: const name = (...) => {...}
        arrow_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*\((.*?)\)\s*=>\s*\{([\s\S]*?)\}"
        matches = re.findall(arrow_pattern, code)
        
        for name, args_str, body in matches:
            args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
            functions.append({
                "name": name,
                "args": args,
                "body": body,
                "type": "arrow_function"
            })
    
    logger.info(f"Extracted {len(functions)} function definitions")
    return functions

def extract_class_definitions(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Extract class definitions from code.
    
    Args:
        code: Code to extract class definitions from
        language: Programming language of the code
        
    Returns:
        List of dictionaries with class information
    """
    logger.info(f"Extracting class definitions for language: {language}")
    
    classes = []
    
    if language.lower() == "python":
        try:
            # Parse the Python code
            tree = ast.parse(code)
            
            # Extract class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Get class name
                    name = node.name
                    
                    # Get base classes
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                    
                    # Get class docstring if available
                    docstring = ast.get_docstring(node)
                    
                    # Get class methods
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append(child.name)
                    
                    # Get class body
                    body_lines = []
                    for line in code.splitlines()[node.lineno:node.end_lineno]:
                        body_lines.append(line)
                    body = "\n".join(body_lines)
                    
                    classes.append({
                        "name": name,
                        "bases": bases,
                        "docstring": docstring,
                        "methods": methods,
                        "body": body,
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno
                    })
        except SyntaxError as e:
            logger.error(f"Syntax error in Python code: {e}")
        except Exception as e:
            logger.error(f"Error extracting class definitions from Python code: {e}")
    elif language.lower() == "javascript":
        # Simple regex-based extraction for JavaScript classes
        # This is a simplified approach and might not handle all cases correctly
        
        # Match class declarations: class Name {...}
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{([\s\S]*?)\}"
        matches = re.findall(class_pattern, code)
        
        for name, base, body in matches:
            # Extract methods using a simple regex
            method_pattern = r"(\w+)\s*\((.*?)\)\s*\{([\s\S]*?)\}"
            method_matches = re.findall(method_pattern, body)
            
            methods = []
            for method_name, _, _ in method_matches:
                if method_name != "constructor":
                    methods.append(method_name)
            
            classes.append({
                "name": name,
                "bases": [base] if base else [],
                "methods": methods,
                "body": body
            })
    
    logger.info(f"Extracted {len(classes)} class definitions")
    return classes

def analyze_code_complexity(code: str, language: str) -> Dict[str, Any]:
    """
    Analyze code complexity.
    
    Args:
        code: Code to analyze
        language: Programming language of the code
        
    Returns:
        Dictionary with complexity metrics
    """
    logger.info(f"Analyzing code complexity for language: {language}")
    
    # Initialize complexity metrics
    metrics = {
        "lines_of_code": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "function_count": 0,
        "class_count": 0,
        "cyclomatic_complexity": 0
    }
    
    # Count lines of code, comment lines, and blank lines
    lines = code.splitlines()
    metrics["lines_of_code"] = len(lines)
    
    for line in lines:
        line = line.strip()
        if not line:
            metrics["blank_lines"] += 1
        elif language.lower() == "python" and (line.startswith("#") or line.startswith('"""') or line.startswith("'''")):
            metrics["comment_lines"] += 1
        elif language.lower() in ["javascript", "typescript"] and (line.startswith("//") or line.startswith("/*") or line.startswith("*")):
            metrics["comment_lines"] += 1
    
    # Count functions and classes
    if language.lower() == "python":
        try:
            tree = ast.parse(code)
            
            # Count functions
            metrics["function_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            
            # Count classes
            metrics["class_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            
            # Calculate cyclomatic complexity (simplified)
            # Count if, for, while, and, or statements
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                    complexity += len(node.values) - 1
            
            metrics["cyclomatic_complexity"] = complexity
        except Exception as e:
            logger.error(f"Error analyzing Python code complexity: {e}")
    elif language.lower() in ["javascript", "typescript"]:
        # Simple regex-based analysis for JavaScript
        # Count functions
        function_count = len(re.findall(r"function\s+\w+\s*\(", code))
        function_count += len(re.findall(r"(?:const|let|var)\s+\w+\s*=\s*(?:function|\(.*?\)\s*=>)", code))
        metrics["function_count"] = function_count
        
        # Count classes
        metrics["class_count"] = len(re.findall(r"class\s+\w+", code))
        
        # Calculate cyclomatic complexity (simplified)
        # Count if, for, while, &&, || statements
        complexity = 0
        complexity += len(re.findall(r"\bif\s*\(", code))
        complexity += len(re.findall(r"\bfor\s*\(", code))
        complexity += len(re.findall(r"\bwhile\s*\(", code))
        complexity += len(re.findall(r"&&", code))
        complexity += len(re.findall(r"\|\|", code))
        
        metrics["cyclomatic_complexity"] = complexity
    
    logger.info(f"Code complexity metrics: {metrics}")
    return metrics

def generate_code_skeleton(
    language: str,
    type: str,
    name: str,
    params: Optional[List[str]] = None,
    return_type: Optional[str] = None,
    description: Optional[str] = None
) -> str:
    """
    Generate a code skeleton for a function or class.
    
    Args:
        language: Programming language for the skeleton
        type: Type of skeleton (function, class, etc.)
        name: Name of the function or class
        params: List of parameter names
        return_type: Return type of the function
        description: Description for the docstring
        
    Returns:
        Generated code skeleton
    """
    logger.info(f"Generating {type} skeleton for {name} in {language}")
    
    params = params or []
    
    if language.lower() == "python":
        if type.lower() == "function":
            # Generate Python function skeleton
            code = f"def {name}({', '.join(params)}):\n"
            
            # Add docstring if description is provided
            if description:
                code += f'    """\n    {description}\n'
                
                # Add parameter descriptions
                if params:
                    code += "\n    Args:\n"
                    for param in params:
                        code += f"        {param}: Description of {param}\n"
                
                # Add return description if return type is provided
                if return_type:
                    code += "\n    Returns:\n"
                    code += f"        {return_type}: Description of return value\n"
                
                code += '    """\n'
            
            # Add function body
            code += "    # TODO: Implement function\n"
            code += "    pass\n"
            
            return code
        elif type.lower() == "class":
            # Generate Python class skeleton
            code = f"class {name}:\n"
            
            # Add docstring if description is provided
            if description:
                code += f'    """\n    {description}\n    """\n\n'
            
            # Add constructor
            code += "    def __init__(self"
            if params:
                code += ", " + ", ".join(params)
            code += "):\n"
            
            # Add constructor body
            if params:
                code += "        # Initialize attributes\n"
                for param in params:
                    code += f"        self.{param} = {param}\n"
            else:
                code += "        # TODO: Initialize attributes\n"
                code += "        pass\n"
            
            return code
    elif language.lower() in ["javascript", "typescript"]:
        if type.lower() == "function":
            # Generate JavaScript function skeleton
            if language.lower() == "typescript" and return_type:
                code = f"function {name}({', '.join(params)}): {return_type} {{\n"
            else:
                code = f"function {name}({', '.join(params)}) {{\n"
            
            # Add function body
            if description:
                code += f"    // {description}\n"
            code += "    // TODO: Implement function\n"
            
            # Add return statement if return type is provided
            if return_type and return_type.lower() != "void":
                if return_type.lower() == "boolean":
                    code += "    return false;\n"
                elif return_type.lower() in ["number", "int", "float"]:
                    code += "    return 0;\n"
                elif return_type.lower() == "string":
                    code += "    return '';\n"
                elif return_type.lower().startswith("array") or return_type.endswith("[]"):
                    code += "    return [];\n"
                elif return_type.lower() == "object":
                    code += "    return {};\n"
                else:
                    code += "    return null;\n"
            
            code += "}\n"
            
            return code
        elif type.lower() == "class":
            # Generate JavaScript class skeleton
            code = f"class {name} {{\n"
            
            # Add constructor
            code += "    constructor("
            if params:
                code += ", ".join(params)
            code += ") {\n"
            
            # Add constructor body
            if description:
                code += f"        // {description}\n"
            
            if params:
                code += "        // Initialize attributes\n"
                for param in params:
                    code += f"        this.{param} = {param};\n"
            else:
                code += "        // TODO: Initialize attributes\n"
            
            code += "    }\n"
            code += "}\n"
            
            return code
    
    # Default case if language or type is not supported
    logger.warning(f"Unsupported language or type: {language}, {type}")
    return f"// TODO: Generate {type} skeleton for {name} in {language}"

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
