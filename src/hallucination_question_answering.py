import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION - Select your LLM provider
# =============================================================================
# Options: "ollama" (default), "anthropic", or "watsonx"
LLM_PROVIDER = "ollama"

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4:latest")

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


def _process_question(args):
    """
    Worker function to generate an answer for a single question.

    This function is designed to be called by multiprocessing.Pool.

    Args:
        args: Tuple of (index, question, provider, model, api_key, watsonx_config)

    Returns:
        Tuple of (index, question, answer)
    """
    index, question, provider, model, api_key, watsonx_config = args

    answer = generate_with_llm(question, provider, model, api_key, watsonx_config)

    return (index, question, answer)


def generate_answers(input_file: Path = None, output_file: Path = None, num_workers: int = None):
    """
    Generate answers for hallucination questions.

    Args:
        input_file: Path to parquet file with 'question' column.
                   If None, uses auto-detected file based on provider.
        output_file: Path to save question-answer pairs.
                    If None, auto-generates based on provider.
        num_workers: Number of parallel workers for multiprocessing.
                    Defaults to cpu_count(). Set to 1 to disable multiprocessing.

    Returns:
        Path to the output parquet file
    """
    # Use default for num_workers if not specified
    if num_workers is None:
        num_workers = cpu_count()
    # Auto-detect input file if not specified
    if input_file is None:
        if LLM_PROVIDER == "anthropic":
            input_file = Path('../data/hallucinationQuestions_anthropic.parquet')
        else:
            input_file = Path(f'../data/hallucinationQuestions_{OLLAMA_MODEL.replace(":", "_")}.parquet')

    # Load questions
    df_questions = pd.read_parquet(input_file, engine='fastparquet')

    if 'question' not in df_questions.columns:
        raise ValueError(f"Input file must contain 'question' column. Found: {df_questions.columns.tolist()}")

    questions = df_questions['question'].tolist()
    print(f"Loaded {len(questions)} questions from {input_file}")
    print(f"Generating answers for each question using {num_workers} worker(s)...")

    # Prepare arguments for each worker
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

    # Prepare WatsonX configuration if using watsonx provider
    watsonx_config = None
    if provider == "watsonx":
        watsonx_config = {
            'base_url': WATSONX_BASE_URL,
            'credentials': WATSONX_BASIC_CREDENTIALS,
            'max_tokens': WATSONX_MAX_TOKENS,
            'token': None  # Will be populated on first use
        }

    worker_args = [(i, question, provider, model, api_key, watsonx_config) for i, question in enumerate(questions)]

    # Process questions in parallel or sequentially
    if num_workers > 1:
        print(f"Processing {len(questions)} questions in parallel...")
        with Pool(num_workers) as pool:
            # Use imap for progress tracking
            results_list = []
            for i, result in enumerate(pool.imap(_process_question, worker_args), 1):
                results_list.append(result)
                if i % 10 == 0:
                    print(f"Generated answers for {i}/{len(questions)} questions")
    else:
        print(f"Processing {len(questions)} questions sequentially...")
        results_list = []
        for i, arg in enumerate(worker_args, 1):
            result = _process_question(arg)
            results_list.append(result)
            if i % 10 == 0:
                print(f"Generated answers for {i}/{len(questions)} questions")

    # Sort results by index and create DataFrame
    results_list.sort(key=lambda x: x[0])
    results = [{'question': q, 'answer': a} for (_idx, q, a) in results_list]

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Auto-generate output filename if not specified
    if output_file is None:
        if LLM_PROVIDER == "anthropic":
            output_file = Path('../data/hallucinationQA_anthropic.parquet')
        elif LLM_PROVIDER == "watsonx":
            output_file = Path(f'../data/hallucinationQA_{WATSONX_MODEL.replace("/", "_").replace(":", "_")}.parquet')
        else:  # ollama
            output_file = Path(f'../data/hallucinationQA_{OLLAMA_MODEL.replace(":", "_")}.parquet')

    # Save results
    df_results.to_parquet(output_file, engine='fastparquet', index=False)

    print(f"\nTotal question-answer pairs generated: {len(results)}")
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Saved to: {output_file}")
    print(f"Columns: {df_results.columns.tolist()}")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate answers for hallucination questions")
    parser.add_argument("--input-file", "-i", type=str, default=None,
                        help="Input parquet file with questions (default: auto-detect based on provider)")
    parser.add_argument("--output-file", "-o", type=str, default=None,
                        help="Output parquet file for Q&A pairs (default: auto-generate based on provider)")
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

    # Convert string paths to Path objects if provided
    input_path = Path(args.input_file) if args.input_file else None
    output_path = Path(args.output_file) if args.output_file else None

    # Generate answers
    output_file = generate_answers(
        input_file=input_path,
        output_file=output_path,
        num_workers=args.workers
    )

    print(f"\n✓ Successfully generated answers: {output_file}")
