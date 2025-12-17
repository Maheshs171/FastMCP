# -*- coding: utf-8 -*-
"""
MCP Server for EVAA Virtual Assistant
Handles tool calls for appointment booking, RAG retrieval, and URL operations
"""

import json
import re
import time
from pinecone import Pinecone
from config import (
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME, 
    FORM_URL_1, 
    FORM_URL_2, 
    cancelAppointmentFormUrl, 
    rescheduleAppointmentFormUrl
)
from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("EVAA Server")

# Initialize Pinecone client
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)


def extract_context_from_message(message: str) -> dict:
    """Extract session_id, path, and bot_id from message format."""
    context = {
        'session_id': None,
        'path': None,
        'bot_id': None,
        'clean_message': message
    }
    
    session_match = re.search(r'\[SESSION_ID:\s*([^\]]+)\]', message)
    if session_match:
        context['session_id'] = session_match.group(1).strip()
    
    path_match = re.search(r'\[PATH:\s*([^\]]+)\]', message)
    if path_match:
        context['path'] = path_match.group(1).strip()
    
    bot_id_match = re.search(r'\[BOT_ID:\s*([^\]]+)\]', message)
    if bot_id_match:
        context['bot_id'] = bot_id_match.group(1).strip()
    
    clean_message = re.sub(r'\[SESSION_ID:[^\]]*\]\s*', '', message)
    clean_message = re.sub(r'\[PATH:[^\]]*\]\s*', '', clean_message)
    clean_message = re.sub(r'\[BOT_ID:[^\]]*\]\s*', '', clean_message)
    context['clean_message'] = clean_message.strip()
    
    return context


def get_bot_id_from_path(path: str) -> str:
    """Extract bot_id from path like 'e1/burneteyecarepinecone/QApixW'"""
    if not path:
        return ""
    return path.strip('/').split('/')[-1]


@mcp.tool()
def book_appointment_tool(query: str = "", session_id: str = None, bot_id: str = None) -> str:
    """Use this tool to open appointment booking form so user can fill form and book appointment."""
    try:
        context = {}
        if query and ('[SESSION_ID:' in query or '[PATH:' in query or '[BOT_ID:' in query):
            context = extract_context_from_message(query)
            bot_id = context.get('bot_id') or bot_id
            session_id = context.get('session_id') or session_id
        
        if not session_id:
            session_id = 'fallback-session-id'
        
        FORM_URL = FORM_URL_2 if bot_id == "fp01" else FORM_URL_1
        separator = '&' if '?' in FORM_URL else '?'
        form_url_with_session = f"{FORM_URL}{separator}session_id={session_id}&ts={int(time.time() * 1000)}"
        
        return json.dumps({
            'reply': 'I\'m opening the appointment booking form for you. Please fill it out to book your appointment.',
            'form_url': form_url_with_session,
            'success': True,
            'session_id': session_id
        })
        
    except Exception as e:
        return json.dumps({
            'reply': 'Sorry, I couldn\'t open the booking form at this moment.',
            'success': False,
            'error': str(e)
        })


@mcp.tool()
def cancel_appointment_tool(query: str = "", session_id: str = None, bot_id: str = None) -> str:
    """Use this tool to open appointment canceling form so user can fill form and cancel appointment."""
    try:
        context = {}
        if query and ('[SESSION_ID:' in query or '[PATH:' in query or '[BOT_ID:' in query):
            context = extract_context_from_message(query)
            bot_id = context.get('bot_id') or bot_id
            session_id = context.get('session_id') or session_id
        
        if not session_id:
            session_id = 'fallback-session-id'
        
        separator = '&' if '?' in cancelAppointmentFormUrl else '?'
        form_url_with_session = f"{cancelAppointmentFormUrl}{separator}session_id={session_id}&ts={int(time.time() * 1000)}"
        
        return json.dumps({
            'reply': 'I\'m opening the appointment cancellation form for you. Please fill it out to cancel your appointment.',
            'form_url': form_url_with_session,
            'success': True,
            'session_id': session_id
        })
        
    except Exception as e:
        return json.dumps({
            'reply': 'Sorry, I couldn\'t open the appointment cancellation form at this moment.',
            'success': False,
            'error': str(e)
        })


@mcp.tool()
def reschedule_appointment_tool(query: str = "", session_id: str = None, bot_id: str = None) -> str:
    """Use this tool to open appointment rescheduling form so user can fill form and reschedule appointment."""
    try:
        context = {}
        if query and ('[SESSION_ID:' in query or '[PATH:' in query or '[BOT_ID:' in query):
            context = extract_context_from_message(query)
            bot_id = context.get('bot_id') or bot_id
            session_id = context.get('session_id') or session_id
        
        if not session_id:
            session_id = 'fallback-session-id'
        
        separator = '&' if '?' in rescheduleAppointmentFormUrl else '?'
        form_url_with_session = f"{rescheduleAppointmentFormUrl}{separator}session_id={session_id}&ts={int(time.time() * 1000)}"
        
        return json.dumps({
            'reply': 'I\'m opening the appointment rescheduling form for you. Please fill it out to reschedule your appointment.',
            'form_url': form_url_with_session,
            'success': True,
            'session_id': session_id
        })
        
    except Exception as e:
        return json.dumps({
            'reply': 'Sorry, I couldn\'t open the appointment rescheduling form at this moment.',
            'success': False,
            'error': str(e)
        })


@mcp.tool()
def rag_retrieval_tool(query: str, session_id: str = None, bot_id: str = None) -> str:
    """
    Use this tool to retrieve relevant information from the vector store for every user query/question.
    Supports dynamic namespace based on bot_id.
    """
    try:
        context = {}
        if query and ('[SESSION_ID:' in query or '[PATH:' in query or '[BOT_ID:' in query):
            context = extract_context_from_message(query)
            session_id = context.get('session_id') or session_id
            bot_id = context.get('bot_id') or bot_id
            query = context.get('clean_message', query)
        
        if not bot_id and context.get('path'):
            bot_id = get_bot_id_from_path(context.get('path'))
        
        namespace = bot_id if bot_id else "hr4s"
        
        ranked_results = index.search_records(
            namespace=namespace,
            query={
                "inputs": {"text": query},
                "top_k": 7
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 5,
                "rank_fields": ["text"]
            },
        )

        results = ranked_results.result.hits
        if not results:
            return "Sorry, I couldn't find any relevant information."
        
        return f"Here's what I found:\n\n{results}"

    except Exception as e:
        return f"Error retrieving context from Pinecone: {str(e)}"
