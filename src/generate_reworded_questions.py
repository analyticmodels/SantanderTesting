import pandas as pd
from pathlib import Path
import random
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION - Select your LLM provider
# =============================================================================
# Options: "ollama" (default) or "anthropic"
LLM_PROVIDER = "anthropic"

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4:latest")

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# Generation settings
TARGET_COUNT = 255   # Number of reworded questions to generate
BATCH_SIZE = 10      # Number of base questions to process at once
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


def generate_with_llm(prompt: str) -> str:
    """Generate a response using the configured LLM provider."""
    if LLM_PROVIDER == "anthropic":
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    else:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()


def generate_reworded_questions(target_count: int = TARGET_COUNT):
    """
    Generate reworded questions that cover the same topics as base questions.

    Args:
        target_count: Number of reworded questions to generate (default: 255)

    Returns:
        Path to the output parquet file
    """
    # Load base questions using fastparquet engine
    base_questions_file = Path('../data/baseQuestions.parquet')
    df_base = pd.read_parquet(base_questions_file, engine='fastparquet')
    base_questions = df_base['question'].tolist()

    print(f"Loaded {len(base_questions)} base questions")
    print(f"Generating {target_count} reworded questions")

    # Generate reworded questions in batches
    reworded_questions = []

    # Shuffle the questions to get variety in order
    shuffled_questions = base_questions.copy()
    random.shuffle(shuffled_questions)

    # Process questions in batches
    num_batches = (target_count + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        # Calculate how many questions to generate in this batch
        remaining = target_count - len(reworded_questions)
        questions_to_generate = min(BATCH_SIZE, remaining)

        if questions_to_generate <= 0:
            break

        # Get a batch of base questions (cycling through if needed)
        start_idx = (batch_idx * BATCH_SIZE) % len(shuffled_questions)
        batch_questions = []
        for i in range(questions_to_generate):
            idx = (start_idx + i) % len(shuffled_questions)
            batch_questions.append(shuffled_questions[idx])

        # Create prompt for batch
        questions_text = '\n'.join([f"{i+1}. {q}" for i, q in enumerate(batch_questions)])

        prompt = f"""Below are {questions_to_generate} questions about a banking service. Your task is to reword each question while keeping the same core topic and meaning, but using different phrasing, word order, and sentence structure.

Original questions:
{questions_text}

Instructions:
- Reword each question to ask about the same topic but with different phrasing
- Use synonyms and different sentence structures
- Maintain the original intent and topic of each question
- Keep questions clear and natural-sounding
- Do not add any hallucination-inducing elements or fabricated details
- Present questions in a different order than the original

Return ONLY the {questions_to_generate} reworded questions, one per line, no numbering or extra text."""

        response_text = generate_with_llm(prompt)

        # Extract questions from response
        generated = response_text.split('\n')
        # Strip leading numbers, periods, dashes, and whitespace from each question
        generated = [re.sub(r'^[\d\.\-\)\s]+', '', q).strip() for q in generated]
        generated = [q for q in generated if q and len(q) > 10]  # Remove empty or very short strings

        reworded_questions.extend(generated[:questions_to_generate])
        print(f"Batch {batch_idx + 1}/{num_batches}: Generated {len(generated[:questions_to_generate])} questions (Total: {len(reworded_questions)}/{target_count})")

    # Ensure we have exactly target_count questions
    reworded_questions = reworded_questions[:target_count]

    print(f"\nTotal reworded questions generated: {len(reworded_questions)}")

    # Create DataFrame and save
    df_reworded = pd.DataFrame({'question': reworded_questions})

    # Set output filename
    output_file = Path('../data/baseQuestions_reworded.parquet')
    df_reworded.to_parquet(output_file, engine='fastparquet', index=False)

    print(f"Saved to: {output_file}")
    print(f"Columns: {df_reworded.columns.tolist()}")
    print(f"\nFirst 5 reworded questions:")
    for i, q in enumerate(reworded_questions[:5]):
        print(f"{i+1}. {q}")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate reworded questions from base questions")
    parser.add_argument("--target-count", "-n", type=int, default=TARGET_COUNT,
                        help=f"Number of reworded questions to generate (default: {TARGET_COUNT})")
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

    # Generate reworded questions
    output_file = generate_reworded_questions(target_count=args.target_count)
