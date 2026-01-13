import pandas as pd
from pathlib import Path
import random
import re
import os
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count
from functools import partial

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION - Select your LLM provider
# =============================================================================
# Options: "ollama" (default), "anthropic", or "watsonx"
LLM_PROVIDER = "ollama"

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# WatsonX settings
WATSONX_BASE_URL = os.getenv("WATSONX_BASE_URL", "https://apigee-outbound-dev1.nonprod.corpint.net")
WATSONX_BASIC_CREDENTIALS = os.getenv(
    "WATSONX_BASIC_CREDENTIALS",
)
WATSONX_MODEL = os.getenv("WATSONX_MODEL", "meta-llama/llama-3-3-70b-instruct")
WATSONX_MAX_TOKENS = int(os.getenv("WATSONX_MAX_TOKENS", "2000"))

# Generation settings
TARGET_COUNT = 50   # Number of hallucination questions to generate
SAMPLE_SIZE = 10    # Number of base questions to sample from
DEFAULT_CONTEXT_FILE = '../data/Openbank_extracted_text.txt'
DEFAULT_BASE_QUESTIONS_FILE = '../data/agent_assist_questions.parquet'
# =============================================================================

# Initialize clients based on provider
anthropic_client = None
if LLM_PROVIDER == "anthropic":
    from anthropic import Anthropic
    import httpx
    http_client = httpx.Client(verify=False)
    anthropic_client = Anthropic(
        api_key=ANTHROPIC_API_KEY,
        http_client=http_client
    )
    print(f"Using Anthropic API with model: {ANTHROPIC_MODEL}")
elif LLM_PROVIDER == "watsonx":
    print(f"Using WatsonX with model: {WATSONX_MODEL}")
    print(f"Base URL: {WATSONX_BASE_URL}")
else:
    import ollama
    print(f"Using Ollama with model: {OLLAMA_MODEL}")


def generate_with_llm(prompt: str, provider: str = None, model: str = None, api_key: str = None,
                     watsonx_config: dict = None) -> str:
    """Generate a response using the configured LLM provider."""
    # Use globals if not specified
    if provider is None:
        provider = LLM_PROVIDER
    if model is None:
        if provider == "anthropic":
            model = ANTHROPIC_MODEL
        elif provider == "watsonx":
            model = WATSONX_MODEL
        else:
            model = OLLAMA_MODEL
    if api_key is None:
        api_key = ANTHROPIC_API_KEY

    if provider == "anthropic":
        from anthropic import Anthropic
        import httpx
        http_client = httpx.Client(verify=False)
        client = Anthropic(api_key=api_key, http_client=http_client)
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    elif provider == "watsonx":
        # WatsonX integration
        try:
            from watsonx import WatsonX
        except ImportError:
            raise ImportError(
                "WatsonX module not found. Please ensure the watsonx module is available."
            )

        # Get watsonx config
        if watsonx_config is None:
            watsonx_config = {
                'base_url': WATSONX_BASE_URL,
                'credentials': WATSONX_BASIC_CREDENTIALS,
                'max_tokens': WATSONX_MAX_TOKENS
            }

        base_url = watsonx_config.get('base_url', WATSONX_BASE_URL)
        credentials = watsonx_config.get('credentials', WATSONX_BASIC_CREDENTIALS)
        max_tokens = watsonx_config.get('max_tokens', WATSONX_MAX_TOKENS)
        token = watsonx_config.get('token')

        # Initialize WatsonX client if needed
        watsonx_client = WatsonX()

        # Get or refresh token
        if token is None:
            oauth_url = f"{base_url}/oauth2/accesstoken-clientcredentials"
            token = watsonx_client.post_oauth2(credentials, oauth_url)

        # Make API call with retry logic for token expiration
        try:
            generated_text, _, _ = watsonx_client.post_text_generation(
                base_url,
                token,
                model,
                prompt,
                max_tokens
            )
            return generated_text.strip()
        except Exception as e:
            # Check if it's a 401 error (token expired)
            if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 401:
                # Refresh token and retry
                oauth_url = f"{base_url}/oauth2/accesstoken-clientcredentials"
                token = watsonx_client.post_oauth2(credentials, oauth_url)
                watsonx_config['token'] = token

                generated_text, _, _ = watsonx_client.post_text_generation(
                    base_url,
                    token,
                    model,
                    prompt,
                    max_tokens
                )
                return generated_text.strip()
            else:
                raise

    else:  # ollama
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()


def _process_base_question(args):
    """
    Worker function to process a single base question.

    This function is designed to be called by multiprocessing.Pool.

    Args:
        args: Tuple of (index, base_question, questions_per_base, context_text, provider, model, api_key, watsonx_config)

    Returns:
        Tuple of (index, list of generated questions)
    """
    index, base_question, questions_per_base, context_text, provider, model, api_key, watsonx_config = args

    # Build prompt with optional context
    context_section = ""
    if context_text:
        context_section = f"""
CONTEXT DOCUMENT:
{context_text}

---
"""

    prompt = f"""{context_section}Based on this question: "{base_question}"

Generate {questions_per_base} new questions designed to cause LLM hallucinations (not just incorrect answers, but actual hallucinations involving confabulation of plausible-sounding but false details).

{f"Your questions should reference the context document above where appropriate." if context_text else ""}

Use these techniques:
- Do not Replace entities
- Add new entities with plausible but potentially non-existent names
- Add specific numerical details (dates, percentages, amounts, limits)
- Combine real concepts with fabricated specifics
- Include precise policy/feature details that require exact knowledge
- Use obscure or ambiguous entity names that sound legitimate
- Entity neighbor contamination
- False presupposition injection
- Cross-Domain jurisdictional query
- Citation or source demand
- combinatorial edge case stacking
- Fabricated authority action
- Negative enumeration request

Use at most three of these techniques per generated question

Return ONLY the questions, one per line, no numbering or extra text."""

    response_text = generate_with_llm(prompt, provider, model, api_key, watsonx_config)

    # Extract questions from response
    generated = response_text.split('\n')
    # Strip leading numbers, periods, dashes, and whitespace from each question
    generated = [re.sub(r'^[\d\.\-\)\s]+', '', q).strip() for q in generated]
    generated = [q for q in generated if q]  # Remove empty strings

    return (index, generated[:questions_per_base])


def generate_hallucination_questions(target_count: int = None, sample_size: int = None,
                                    context_file: str = None, base_questions_file: str = None,
                                    num_workers: int = None):
    """
    Generate hallucination-inducing questions.

    Args:
        target_count: Number of hallucination questions to generate.
                     Defaults to TARGET_COUNT from config.
        sample_size: Number of base questions to sample from.
                    Defaults to SAMPLE_SIZE from config.
        context_file: Path to text file containing context for question generation.
                     Defaults to DEFAULT_CONTEXT_FILE.
        base_questions_file: Path to parquet file containing base questions.
                            Defaults to DEFAULT_BASE_QUESTIONS_FILE.
        num_workers: Number of parallel workers for multiprocessing.
                    Defaults to cpu_count(). Set to 1 to disable multiprocessing.

    Returns:
        Path to the output parquet file containing generated questions
    """
    # Use defaults from config if not specified
    if target_count is None:
        target_count = TARGET_COUNT
    if sample_size is None:
        sample_size = SAMPLE_SIZE
    if context_file is None:
        context_file = DEFAULT_CONTEXT_FILE
    if base_questions_file is None:
        base_questions_file = DEFAULT_BASE_QUESTIONS_FILE
    if num_workers is None:
        num_workers = cpu_count()

    # Load context from file
    context_text = ""
    try:
        with open(context_file, 'r', encoding='utf-8') as f:
            context_text = f.read()
        print(f"Loaded context from: {context_file} ({len(context_text)} characters)")
    except FileNotFoundError:
        print(f"Warning: Context file not found at {context_file}. Proceeding without context.")
    except Exception as e:
        print(f"Warning: Error reading context file: {e}. Proceeding without context.")

    # Load base questions using fastparquet engine
    base_questions_path = Path(base_questions_file)
    df_base = pd.read_parquet(base_questions_path, engine='fastparquet')
    base_questions = df_base['question'].tolist()

    print(f"Loaded {len(base_questions)} base questions from {base_questions_path}")
    print(f"Generating {target_count} questions from {sample_size} base samples")
    print(f"Using {num_workers} worker(s) for parallel processing")

    # Generate hallucination-inducing questions
    sample_size = min(sample_size, len(base_questions))
    sampled_questions = random.sample(base_questions, sample_size)

    # Prepare arguments for each worker
    worker_args = []

    # Prepare WatsonX configuration if using watsonx provider
    watsonx_config = None
    if LLM_PROVIDER == "watsonx":
        watsonx_config = {
            'base_url': WATSONX_BASE_URL,
            'credentials': WATSONX_BASIC_CREDENTIALS,
            'max_tokens': WATSONX_MAX_TOKENS,
            'token': None  # Will be populated on first use
        }

    for i, base_question in enumerate(sampled_questions):
        # Calculate how many questions this base question should generate
        questions_per_base = (target_count // sample_size) + (1 if i < (target_count % sample_size) else 0)

        # Get LLM configuration
        provider = LLM_PROVIDER
        if provider == "anthropic":
            model = ANTHROPIC_MODEL
            api_key = ANTHROPIC_API_KEY
        elif provider == "watsonx":
            model = WATSONX_MODEL
            api_key = None
        else:  # ollama
            model = OLLAMA_MODEL
            api_key = None

        worker_args.append((i, base_question, questions_per_base, context_text, provider, model, api_key, watsonx_config))

    # Process questions in parallel or sequentially
    if num_workers > 1:
        print(f"Processing {len(worker_args)} base questions in parallel...")
        with Pool(num_workers) as pool:
            results = pool.map(_process_base_question, worker_args)
    else:
        print(f"Processing {len(worker_args)} base questions sequentially...")
        results = [_process_base_question(arg) for arg in worker_args]

    # Sort results by index and collect all questions
    results.sort(key=lambda x: x[0])
    hallucination_questions = []
    for i, (_idx, questions) in enumerate(results):
        hallucination_questions.extend(questions)
        print(f"Generated {len(questions)} questions from base question {i+1}/{len(sampled_questions)}")

    # Ensure we have exactly target_count questions
    hallucination_questions = hallucination_questions[:target_count]

    print(f"\nTotal hallucination questions generated: {len(hallucination_questions)}")

    # Create DataFrame and save
    df_hallucination = pd.DataFrame({'question': hallucination_questions})

    # Set output filename based on provider
    if LLM_PROVIDER == "anthropic":
        output_file = Path('../data/hallucinationQuestions_anthropic.parquet')
    elif LLM_PROVIDER == "watsonx":
        output_file = Path(f'../data/hallucinationQuestions_{WATSONX_MODEL.replace("/", "_").replace(":", "_")}.parquet')
    else:  # ollama
        output_file = Path(f'../data/hallucinationQuestions_{OLLAMA_MODEL.replace(":", "_")}.parquet')

    df_hallucination.to_parquet(output_file, engine='fastparquet', index=False)

    print(f"Provider: {LLM_PROVIDER}")
    print(f"Saved questions to: {output_file}")
    print(f"Columns: {df_hallucination.columns.tolist()}")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate hallucination-inducing questions")
    parser.add_argument("--target-count", "-n", type=int, default=TARGET_COUNT,
                        help=f"Number of hallucination questions to generate (default: {TARGET_COUNT})")
    parser.add_argument("--sample-size", "-s", type=int, default=SAMPLE_SIZE,
                        help=f"Number of base questions to sample from (default: {SAMPLE_SIZE})")
    parser.add_argument("--context-file", "-c", type=str, default=None,
                        help=f"Path to context file for question generation (default: {DEFAULT_CONTEXT_FILE})")
    parser.add_argument("--base-questions", "-b", type=str, default=None,
                        help=f"Path to base questions parquet file (default: {DEFAULT_BASE_QUESTIONS_FILE})")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help=f"Number of parallel workers (default: {cpu_count()}). Set to 1 to disable multiprocessing")
    parser.add_argument("--provider", "-p", choices=["ollama", "anthropic", "watsonx"], default=None,
                        help="LLM provider to use (default: from config)")

    args = parser.parse_args()

    # Override provider if specified
    if args.provider:
        LLM_PROVIDER = args.provider
        if LLM_PROVIDER == "anthropic":
            from anthropic import Anthropic
            import httpx
            http_client = httpx.Client(verify=False)
            anthropic_client = Anthropic(
                api_key=ANTHROPIC_API_KEY,
                http_client=http_client
            )
            print(f"Switched to Anthropic API with model: {ANTHROPIC_MODEL}")
        elif LLM_PROVIDER == "watsonx":
            print(f"Switched to WatsonX with model: {WATSONX_MODEL}")
            print(f"Base URL: {WATSONX_BASE_URL}")

    # Generate hallucination questions
    output_file = generate_hallucination_questions(
        target_count=args.target_count,
        sample_size=args.sample_size,
        context_file=args.context_file,
        base_questions_file=args.base_questions,
        num_workers=args.workers
    )

    print(f"\n✓ Successfully generated questions: {output_file}")
