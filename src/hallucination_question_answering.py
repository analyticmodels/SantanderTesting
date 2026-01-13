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
# Options: "ollama" (default) or "anthropic"
LLM_PROVIDER = "ollama"

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4:latest")

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
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
else:
    import ollama
    print(f"Using Ollama with model: {OLLAMA_MODEL}")


def generate_with_llm(prompt: str, provider: str = None, model: str = None, api_key: str = None) -> str:
    """Generate a response using the configured LLM provider."""
    # Use globals if not specified
    if provider is None:
        provider = LLM_PROVIDER
    if model is None:
        model = ANTHROPIC_MODEL if provider == "anthropic" else OLLAMA_MODEL
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
    else:
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
        args: Tuple of (index, question, provider, model, api_key)

    Returns:
        Tuple of (index, question, answer)
    """
    index, question, provider, model, api_key = args

    answer = generate_with_llm(question, provider, model, api_key)

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
    model = ANTHROPIC_MODEL if provider == "anthropic" else OLLAMA_MODEL
    api_key = ANTHROPIC_API_KEY if provider == "anthropic" else None

    worker_args = [(i, question, provider, model, api_key) for i, question in enumerate(questions)]

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
        else:
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
    parser.add_argument("--provider", "-p", choices=["ollama", "anthropic"], default=None,
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
