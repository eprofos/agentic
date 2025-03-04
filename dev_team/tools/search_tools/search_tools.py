"""
Search tools for the Pydantic AI agent system.
These tools help with searching for information online.
"""

import re
import json
from typing import List, Dict, Any, Optional, Union
from urllib.parse import quote

from utils.logging.logger import setup_logger

# Create a logger for this module
logger = setup_logger("search_tools")

def mock_web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Mock web search function that returns simulated search results.
    In a real implementation, this would use an actual search API.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of dictionaries with search results
    """
    logger.info(f"Performing mock web search for query: {query}")
    
    # Normalize the query
    query = query.lower()
    
    # Define some mock search results based on common programming topics
    mock_results = {
        "python": [
            {
                "title": "Python Documentation",
                "url": "https://docs.python.org/3/",
                "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively."
            },
            {
                "title": "Getting Started with Python",
                "url": "https://www.python.org/about/gettingstarted/",
                "snippet": "Python is a powerful programming language that's easy to learn and fun to play with."
            },
            {
                "title": "Python Tutorial - W3Schools",
                "url": "https://www.w3schools.com/python/",
                "snippet": "Python is a popular programming language. It was created by Guido van Rossum, and released in 1991."
            }
        ],
        "javascript": [
            {
                "title": "JavaScript - MDN Web Docs",
                "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
                "snippet": "JavaScript (JS) is a lightweight, interpreted, or just-in-time compiled programming language with first-class functions."
            },
            {
                "title": "Learn JavaScript - Codecademy",
                "url": "https://www.codecademy.com/learn/introduction-to-javascript",
                "snippet": "JavaScript is a powerful, flexible, and fast programming language now being used for increasingly complex web development."
            },
            {
                "title": "JavaScript Tutorial - W3Schools",
                "url": "https://www.w3schools.com/js/",
                "snippet": "JavaScript is the world's most popular programming language. JavaScript is the programming language of the Web."
            }
        ],
        "react": [
            {
                "title": "React – A JavaScript library for building user interfaces",
                "url": "https://reactjs.org/",
                "snippet": "React makes it painless to create interactive UIs. Design simple views for each state in your application."
            },
            {
                "title": "Getting Started – React",
                "url": "https://reactjs.org/docs/getting-started.html",
                "snippet": "This page is an overview of the React documentation and related resources."
            },
            {
                "title": "React Tutorial - W3Schools",
                "url": "https://www.w3schools.com/react/",
                "snippet": "React is a JavaScript library created by Facebook. React is a User Interface (UI) library."
            }
        ],
        "machine learning": [
            {
                "title": "Machine Learning - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Machine_learning",
                "snippet": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'."
            },
            {
                "title": "Machine Learning | Coursera",
                "url": "https://www.coursera.org/learn/machine-learning",
                "snippet": "Machine learning is the science of getting computers to act without being explicitly programmed."
            },
            {
                "title": "Machine Learning - Stanford University | Coursera",
                "url": "https://www.coursera.org/learn/machine-learning-course",
                "snippet": "This course provides a broad introduction to machine learning, datamining, and statistical pattern recognition."
            }
        ],
        "api": [
            {
                "title": "API - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/API",
                "snippet": "An application programming interface (API) is a connection between computers or between computer programs."
            },
            {
                "title": "What is an API? (Application Programming Interface) | MuleSoft",
                "url": "https://www.mulesoft.com/resources/api/what-is-an-api",
                "snippet": "API is the acronym for Application Programming Interface, which is a software intermediary that allows two applications to talk to each other."
            },
            {
                "title": "What is an API? | IBM",
                "url": "https://www.ibm.com/cloud/learn/api",
                "snippet": "An API, or application programming interface, is a set of defined rules that enable different applications to communicate with each other."
            }
        ],
        "database": [
            {
                "title": "Database - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Database",
                "snippet": "A database is an organized collection of data, generally stored and accessed electronically from a computer system."
            },
            {
                "title": "What is a Database? | Oracle",
                "url": "https://www.oracle.com/database/what-is-database/",
                "snippet": "A database is an organized collection of structured information, or data, typically stored electronically in a computer system."
            },
            {
                "title": "Database Definition & Meaning - Merriam-Webster",
                "url": "https://www.merriam-webster.com/dictionary/database",
                "snippet": "Database: a usually large collection of data organized especially for rapid search and retrieval (as by a computer)."
            }
        ]
    }
    
    # Find the best matching topic
    best_match = None
    best_match_score = 0
    
    for topic in mock_results:
        if topic in query:
            # If the topic is a direct substring of the query, use it
            best_match = topic
            break
        else:
            # Calculate a simple similarity score based on word overlap
            topic_words = set(topic.split())
            query_words = set(query.split())
            overlap = len(topic_words.intersection(query_words))
            
            if overlap > best_match_score:
                best_match_score = overlap
                best_match = topic
    
    # If no good match was found, use a generic response
    if best_match is None or best_match_score == 0:
        return [
            {
                "title": "Search results for: " + query,
                "url": f"https://www.google.com/search?q={quote(query)}",
                "snippet": "No specific information available for this query. Try refining your search terms."
            }
        ]
    
    # Return the mock results for the best matching topic
    results = mock_results.get(best_match, [])
    
    # Add a generic result with the actual query
    results.append({
        "title": f"Search results for: {query}",
        "url": f"https://www.google.com/search?q={quote(query)}",
        "snippet": f"More results for {query} can be found by following this link."
    })
    
    # Limit the number of results
    return results[:num_results]

def mock_documentation_search(query: str, language: Optional[str] = None, framework: Optional[str] = None, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Mock documentation search function that returns simulated documentation results.
    In a real implementation, this would use an actual documentation search API.
    
    Args:
        query: Search query
        language: Programming language to search documentation for
        framework: Framework to search documentation for
        num_results: Number of results to return
        
    Returns:
        List of dictionaries with documentation search results
    """
    logger.info(f"Performing mock documentation search for query: {query}, language: {language}, framework: {framework}")
    
    # Normalize inputs
    query = query.lower()
    language = language.lower() if language else None
    framework = framework.lower() if framework else None
    
    # Define some mock documentation results based on common programming topics
    mock_results = {
        "python": {
            "general": [
                {
                    "title": "Built-in Functions - Python Documentation",
                    "url": "https://docs.python.org/3/library/functions.html",
                    "snippet": "The Python interpreter has a number of functions and types built into it that are always available."
                },
                {
                    "title": "The Python Standard Library - Python Documentation",
                    "url": "https://docs.python.org/3/library/index.html",
                    "snippet": "While The Python Language Reference describes the exact syntax and semantics of the Python language, this library reference manual describes the standard library that is distributed with Python."
                },
                {
                    "title": "Python HOWTOs - Python Documentation",
                    "url": "https://docs.python.org/3/howto/index.html",
                    "snippet": "Python HOWTOs are documents that cover a single, specific topic, and attempt to cover it fairly completely."
                }
            ],
            "django": [
                {
                    "title": "Django Documentation",
                    "url": "https://docs.djangoproject.com/en/stable/",
                    "snippet": "Django is a high-level Python Web framework that encourages rapid development and clean, pragmatic design."
                },
                {
                    "title": "Django Models - Django Documentation",
                    "url": "https://docs.djangoproject.com/en/stable/topics/db/models/",
                    "snippet": "A model is the single, definitive source of information about your data. It contains the essential fields and behaviors of the data you're storing."
                },
                {
                    "title": "Django Views - Django Documentation",
                    "url": "https://docs.djangoproject.com/en/stable/topics/http/views/",
                    "snippet": "A view function, or view for short, is a Python function that takes a Web request and returns a Web response."
                }
            ],
            "flask": [
                {
                    "title": "Flask Documentation",
                    "url": "https://flask.palletsprojects.com/en/latest/",
                    "snippet": "Flask is a lightweight WSGI web application framework. It is designed to make getting started quick and easy, with the ability to scale up to complex applications."
                },
                {
                    "title": "Flask Quickstart - Flask Documentation",
                    "url": "https://flask.palletsprojects.com/en/latest/quickstart/",
                    "snippet": "Eager to get started? This page gives a good introduction to Flask. It assumes you already have Flask installed."
                },
                {
                    "title": "Flask Tutorial - Flask Documentation",
                    "url": "https://flask.palletsprojects.com/en/latest/tutorial/",
                    "snippet": "This tutorial will walk you through creating a basic blog application called Flaskr."
                }
            ]
        },
        "javascript": {
            "general": [
                {
                    "title": "JavaScript Guide - MDN Web Docs",
                    "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
                    "snippet": "The JavaScript Guide shows you how to use JavaScript and gives an overview of the language."
                },
                {
                    "title": "JavaScript Reference - MDN Web Docs",
                    "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference",
                    "snippet": "This part of the JavaScript section on MDN serves as a repository of facts about the JavaScript language."
                },
                {
                    "title": "JavaScript Standard Built-in Objects - MDN Web Docs",
                    "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects",
                    "snippet": "This chapter documents all of JavaScript's standard, built-in objects, including their methods and properties."
                }
            ],
            "react": [
                {
                    "title": "React Documentation",
                    "url": "https://reactjs.org/docs/getting-started.html",
                    "snippet": "This page is an overview of the React documentation and related resources."
                },
                {
                    "title": "React Components - React Documentation",
                    "url": "https://reactjs.org/docs/components-and-props.html",
                    "snippet": "Components let you split the UI into independent, reusable pieces, and think about each piece in isolation."
                },
                {
                    "title": "React Hooks - React Documentation",
                    "url": "https://reactjs.org/docs/hooks-intro.html",
                    "snippet": "Hooks are a new addition in React 16.8. They let you use state and other React features without writing a class."
                }
            ],
            "vue": [
                {
                    "title": "Vue.js Documentation",
                    "url": "https://vuejs.org/guide/introduction.html",
                    "snippet": "Vue (pronounced /vjuː/, like view) is a JavaScript framework for building user interfaces."
                },
                {
                    "title": "Vue.js Components - Vue.js Documentation",
                    "url": "https://vuejs.org/guide/essentials/component-basics.html",
                    "snippet": "Components are reusable Vue instances with a name. We can use this component as a custom element inside a root Vue instance."
                },
                {
                    "title": "Vue.js Reactivity - Vue.js Documentation",
                    "url": "https://vuejs.org/guide/essentials/reactivity-fundamentals.html",
                    "snippet": "One of Vue's most distinctive features is its unobtrusive reactivity system."
                }
            ]
        }
    }
    
    # Determine which results to return based on language and framework
    if language and language in mock_results:
        if framework and framework in mock_results[language]:
            results = mock_results[language][framework]
        else:
            results = mock_results[language]["general"]
    else:
        # If language is not specified or not found, return a generic response
        return [
            {
                "title": "Documentation search for: " + query,
                "url": f"https://devdocs.io/#q={quote(query)}",
                "snippet": "DevDocs combines multiple API documentations in a fast, organized, and searchable interface."
            }
        ]
    
    # Add a generic result with the actual query
    results.append({
        "title": f"Documentation search for: {query}",
        "url": f"https://devdocs.io/#q={quote(query)}",
        "snippet": f"More documentation for {query} can be found by following this link."
    })
    
    # Limit the number of results
    return results[:num_results]

def extract_code_from_search_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract code snippets from search results.
    
    Args:
        results: List of search results
        
    Returns:
        List of dictionaries with code snippets
    """
    logger.info("Extracting code snippets from search results")
    
    code_snippets = []
    
    # Regular expression to match code blocks in markdown or HTML
    code_patterns = [
        r"```(\w*)\n([\s\S]*?)\n```",  # Markdown code blocks
        r"<pre><code(?:\s+class=\"(\w+)\")?>([^<]+)</code></pre>",  # HTML code blocks
        r"<pre>([^<]+)</pre>"  # Simple pre tags
    ]
    
    for result in results:
        snippet = result.get("snippet", "")
        
        for pattern in code_patterns:
            matches = re.findall(pattern, snippet)
            
            for match in matches:
                if len(match) == 2:
                    language, code = match
                else:
                    language = "text"
                    code = match[0]
                
                code_snippets.append({
                    "language": language or "text",
                    "code": code.strip(),
                    "source": result.get("title", "Unknown"),
                    "url": result.get("url", "")
                })
    
    logger.info(f"Extracted {len(code_snippets)} code snippets from search results")
    return code_snippets

def summarize_search_results(results: List[Dict[str, str]], query: str) -> str:
    """
    Generate a summary of search results.
    
    Args:
        results: List of search results
        query: Original search query
        
    Returns:
        Summary of search results
    """
    logger.info(f"Summarizing search results for query: {query}")
    
    if not results:
        return f"No results found for query: {query}"
    
    # In a real implementation, this might use an LLM to generate a summary
    # For now, we'll create a simple summary
    
    summary = f"Search results for: {query}\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("snippet", "No description available")
        
        summary += f"{i}. {title}\n"
        summary += f"   URL: {url}\n"
        summary += f"   {snippet}\n\n"
    
    logger.info("Generated search results summary")
    return summary
