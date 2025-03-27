# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/2023_Obsidian_logo.svg/langfr-120px-2023_Obsidian_logo.svg.png" width="30" alt="Obsidian Logo"> Custom Obsidian AI

Run local AI models and use them to extract content from a web page using the Obsidian Clipper plugin.

## Project Structure

The project follows a modular structure with separation of concerns:

```
obsidian_api/
├── app/
│   ├── __init__.py
│   ├── main.py               # Application entry point
│   ├── api/                  # API routes
│   │   ├── __init__.py
│   │   ├── api_v1/
│   │   │   ├── __init__.py
│   │   │   ├── api.py        # Main API router
│   │   │   └── endpoints/    # Endpoint modules
│   │   │       ├── __init__.py
│   │   │       ├── models.py # Model endpoints
│   │   │       └── completion.py # OpenAI-compatible endpoints
│   ├── core/                 # Core application code
│   │   ├── __init__.py
│   │   └── config.py         # Application settings
│   ├── services/             # Business logic services
│   │   ├── __init__.py
│   │   ├── model_service.py  # Model loading service
│   │   └── completion_service.py # Text generation service
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── dependencies.py   # FastAPI dependencies
├── prompts/                  # Prompt templates
│   ├── system.txt            # System prompt template
│   └── user.txt              # User prompt template
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Virtual Environment

1. Install virtualenv:
   ```bash
   pip install virtualenv
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   ```

3. Activate the virtual environment:
   - On Linux/Mac:
     ```bash
     source env/bin/activate
     ```
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```

### Installation

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Optional: Create a `.env` file in the root directory for environment variables:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

### Pre-commit Hooks

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Set up the pre-commit hooks:
   ```bash
   pre-commit install
   ```

   **Note**: You can skip the pre-commit validation using `-n`:
   ```bash
   git commit -m 'my_message' -n
   ```

### Running the Application

```bash
uvicorn app.main:app --reload
```

The application will be available at `http://localhost:8000`.

## API Documentation

Once the application is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Hugging Face Model Loading

```
GET /api/v1/models/{model_name}
```

Loads a model from Hugging Face and optionally processes text with it.

**Query Parameters:**
- `text` (optional): Text to process with the model
- `use_cache` (optional, default: true): Whether to use cached model if available

**Example Request:**
```
GET /api/v1/models/bert-base-uncased?text=Hello%20world
```

### OpenAI-Compatible Chat Completion

```
POST /api/v1/chat/completions
```

This endpoint is compatible with the OpenAI Chat Completions API but uses Hugging Face models under the hood.

**Request Body:**
```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant..."
    },
    {
      "role": "user",
      "content": "Your markdown content from the webpage here..."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

**Response:**
```json
{
  "id": "chatcmpl-123abc...",
  "object": "chat.completion",
  "created": 1679358384,
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The generated response..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 124,
    "completion_tokens": 345,
    "total_tokens": 469
  }
}
```

### Clear Model Cache

```
GET /api/v1/models/clear-cache
```

Clears the model cache to free up memory.

## Environment Variables

- `HUGGINGFACE_TOKEN`: Optional token for accessing private models on Hugging Face
- `DEBUG`: Set to `False` in production to disable auto-reload
- `PORT`: Server port (default: 8000)

## Customizing Prompts

The system and user prompts can be customized by editing the files in the `prompts/` directory:

- `system.txt`: Defines the system instructions for the AI
- `user.txt`: Template for processing user content (uses `{content}` placeholder)
