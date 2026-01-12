import pandas as pd
from pathlib import Path
import random
import json
import re
import os
from dotenv import load_dotenv

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

# Generation settings
TARGET_COUNT = 50   # Number of hallucination questions to generate
SAMPLE_SIZE = 10    # Number of base questions to sample from

# Hallucination detection settings
# Options: "lynx" (Patronus Lynx - specialized) or "llama" (Llama 3.2 - general purpose)
DETECTION_MODEL = os.getenv("DETECTION_MODEL", "lynx")

# Lynx detection settings
LYNX_MODEL = os.getenv("LYNX_MODEL", "tensortemplar/patronus-lynx:8b-instruct-q4_k_m")

# Llama 3.2 detection settings
LLAMA_DETECTION_MODEL = os.getenv("LLAMA_DETECTION_MODEL", "llama3.2:latest")
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
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    else:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()


def generate_hallucination_questions(target_count: int = None, sample_size: int = None):
    """
    Generate hallucination-inducing questions and their answers.

    Args:
        target_count: Number of hallucination questions to generate.
                     Defaults to TARGET_COUNT from config.
        sample_size: Number of base questions to sample from.
                    Defaults to SAMPLE_SIZE from config.

    Returns:
        Path to the output parquet file
    """
    # Use defaults from config if not specified
    if target_count is None:
        target_count = TARGET_COUNT
    if sample_size is None:
        sample_size = SAMPLE_SIZE

    # Load base questions using fastparquet engine
    base_questions_file = Path('../data/baseQuestions.parquet')
    df_base = pd.read_parquet(base_questions_file, engine='fastparquet')
    base_questions = df_base['question'].tolist()

    print(f"Loaded {len(base_questions)} base questions")
    print(f"Generating {target_count} questions from {sample_size} base samples")

    # Generate hallucination-inducing questions
    hallucination_questions = []
    sample_size = min(sample_size, len(base_questions))
    sampled_questions = random.sample(base_questions, sample_size)

    for i, base_question in enumerate(sampled_questions):
        # Generate multiple variations per base question
        questions_per_base = (target_count // sample_size) + (1 if i < (target_count % sample_size) else 0)

        prompt = f"""Based on this question: "{base_question}"

Generate {questions_per_base} new questions designed to cause LLM hallucinations (not just incorrect answers, but actual hallucinations involving confabulation of plausible-sounding but false details).

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

        response_text = generate_with_llm(prompt)

        # Extract questions from response
        generated = response_text.split('\n')
        # Strip leading numbers, periods, dashes, and whitespace from each question
        generated = [re.sub(r'^[\d\.\-\)\s]+', '', q).strip() for q in generated]
        generated = [q for q in generated if q]  # Remove empty strings

        hallucination_questions.extend(generated[:questions_per_base])
        print(f"Generated {len(generated[:questions_per_base])} questions from base question {i+1}/{len(sampled_questions)}")

    # Ensure we have exactly target_count questions
    hallucination_questions = hallucination_questions[:target_count]

    print(f"\nTotal hallucination questions generated: {len(hallucination_questions)}")
    print("Now generating answers for each question...")

    # Generate answers for each hallucination question
    results = []
    for i, question in enumerate(hallucination_questions):
        answer = generate_with_llm(question)
        results.append({
            'question': question,
            'answer': answer
        })

        if (i + 1) % 10 == 0:
            print(f"Generated answers for {i + 1}/{len(hallucination_questions)} questions")

    # Create DataFrame and save
    df_hallucination = pd.DataFrame(results)

    # Set output filename based on provider
    if LLM_PROVIDER == "anthropic":
        output_file = Path('../data/hallucinationQuestions_anthropic.parquet')
    else:
        output_file = Path('../data/hallucinationQuestions_gemma3.parquet')

    df_hallucination.to_parquet(output_file, engine='fastparquet', index=False)

    print(f"\nTotal question-answer pairs generated: {len(results)}")
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Saved to: {output_file}")
    print(f"Columns: {df_hallucination.columns.tolist()}")

    return output_file


# =============================================================================
# Patronus Lynx Hallucination Detection
# =============================================================================

# Lynx prompt template for hallucination detection
LYNX_PROMPT_TEMPLATE = """Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT.

The ANSWER must not offer new information beyond the context provided in the DOCUMENT.

The ANSWER also must not contradict information provided in the DOCUMENT.

Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT.

Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{document}

--
ANSWER:
{answer}

--
Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
"""

# Llama 3.2 prompt template for hallucination detection
LLAMA_HALLUCINATION_PROMPT = """You are an expert hallucination detector for AI-generated text. Your task is to determine whether an ANSWER contains hallucinations when evaluated against a REFERENCE DOCUMENT.

## Definition of Hallucination
A hallucination occurs when the AI generates content that is:
1. **Fabricated**: Information that is made up and not present in the reference document
2. **Contradictory**: Information that directly conflicts with the reference document
3. **Unsupported**: Specific claims (names, numbers, dates, facts) that cannot be verified from the reference
4. **Confabulated**: Plausible-sounding details that fill gaps with invented specifics

## What is NOT a Hallucination
- Reasonable inferences directly supported by the document
- Paraphrasing or summarizing document content accurately
- General knowledge that doesn't conflict with the document
- Appropriate hedging language ("may", "could", "typically")

## Your Task
Analyze the ANSWER for hallucinations by checking each claim against the REFERENCE DOCUMENT.

---
QUESTION:
{question}

---
REFERENCE DOCUMENT:
{document}

---
ANSWER TO EVALUATE:
{answer}

---
## Evaluation Instructions
1. Identify each factual claim in the ANSWER
2. For each claim, determine if it is:
   - SUPPORTED: Directly stated or clearly implied by the document
   - UNSUPPORTED: Not mentioned in the document (potential hallucination)
   - CONTRADICTED: Conflicts with the document (definite hallucination)
3. Consider the severity: Minor unsupported details vs. major fabricated facts

## Output Format
Respond with valid JSON only, no additional text:
{{
    "REASONING": [
        "Claim 1: <claim> - <SUPPORTED/UNSUPPORTED/CONTRADICTED> - <explanation>",
        "Claim 2: <claim> - <SUPPORTED/UNSUPPORTED/CONTRADICTED> - <explanation>"
    ],
    "HALLUCINATION_DETECTED": <true or false>,
    "CONFIDENCE": "<HIGH/MEDIUM/LOW>",
    "SCORE": "<PASS if no hallucination, FAIL if hallucination detected>"
}}
"""

# Llama 3.2 prompt for cases without reference document
LLAMA_HALLUCINATION_PROMPT_NO_DOC = """You are an expert hallucination detector for AI-generated text. Your task is to identify potential hallucinations in an ANSWER when NO reference document is available.

## Definition of Hallucination (Without Reference)
Without a reference document, focus on detecting:
1. **Overly Specific Fabrications**: Suspiciously precise details (exact dates, percentages, names) that seem invented
2. **False Authority Claims**: References to non-existent studies, regulations, or sources
3. **Confident Misinformation**: Stating uncertain or likely false information as definite fact
4. **Logical Inconsistencies**: Internal contradictions within the answer itself
5. **Implausible Claims**: Information that defies common knowledge or basic logic

## Indicators of Potential Hallucination
- Very specific numbers/dates without hedging (e.g., "exactly 47.3%" vs "approximately 50%")
- Named sources that cannot be easily verified
- Claims presented with high confidence on obscure or complex topics
- Technical details that seem too convenient or complete

## Your Task
Analyze the ANSWER for signs of hallucination based on internal consistency and plausibility.

---
QUESTION:
{question}

---
ANSWER TO EVALUATE:
{answer}

---
## Evaluation Instructions
1. Identify claims that appear suspiciously specific or authoritative
2. Check for internal logical consistency
3. Assess whether confidence level matches the verifiability of claims
4. Flag any claims that seem fabricated or implausible

## Output Format
Respond with valid JSON only, no additional text:
{{
    "REASONING": [
        "Observation 1: <specific concern or observation>",
        "Observation 2: <specific concern or observation>"
    ],
    "SUSPICIOUS_CLAIMS": [
        "<list any claims that appear potentially hallucinated>"
    ],
    "HALLUCINATION_DETECTED": <true or false>,
    "CONFIDENCE": "<HIGH/MEDIUM/LOW>",
    "SCORE": "<PASS if likely faithful, FAIL if hallucination likely>"
}}
"""


def detect_hallucination_lynx(question: str, answer: str, document: str = "") -> dict:
    """
    Use Patronus Lynx to detect if an answer contains hallucinations.

    Args:
        question: The question that was asked
        answer: The answer to evaluate
        document: Reference document/context (if empty, we're checking for unsupported claims)

    Returns:
        dict with 'reasoning', 'score', 'confidence', and 'raw_response'
    """
    import ollama as ollama_client

    # If no document provided, use a minimal context indicating no ground truth available
    if not document:
        document = "No reference document provided. Evaluate if the answer makes specific claims that cannot be verified."

    prompt = LYNX_PROMPT_TEMPLATE.format(
        question=question,
        document=document,
        answer=answer
    )

    response = ollama_client.chat(
        model=LYNX_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response['message']['content'].strip()

    # Try to parse JSON from response
    try:
        # Find JSON in response (it might have extra text)
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            parsed = json.loads(json_str)
            return {
                'reasoning': parsed.get('REASONING', ''),
                'score': parsed.get('SCORE', 'UNKNOWN'),
                'confidence': 'HIGH',  # Lynx is specialized, assume high confidence
                'raw_response': raw_response
            }
    except json.JSONDecodeError:
        pass

    # Fallback: extract score from text
    score = 'UNKNOWN'
    if 'FAIL' in raw_response.upper():
        score = 'FAIL'
    elif 'PASS' in raw_response.upper():
        score = 'PASS'

    return {
        'reasoning': raw_response,
        'score': score,
        'confidence': 'LOW',  # Couldn't parse structured response
        'raw_response': raw_response
    }


def detect_hallucination_llama(question: str, answer: str, document: str = "") -> dict:
    """
    Use Llama 3.2 to detect if an answer contains hallucinations.

    This function uses a general-purpose LLM with a carefully crafted prompt
    to identify hallucinations. It works in two modes:
    - With document: Checks if answer is faithful to the reference
    - Without document: Checks for suspicious fabrications and inconsistencies

    Args:
        question: The question that was asked
        answer: The answer to evaluate
        document: Reference document/context (optional)

    Returns:
        dict with 'reasoning', 'score', 'confidence', 'suspicious_claims', and 'raw_response'
    """
    import ollama as ollama_client

    # Select appropriate prompt based on whether document is provided
    if document and document.strip():
        prompt = LLAMA_HALLUCINATION_PROMPT.format(
            question=question,
            document=document,
            answer=answer
        )
    else:
        prompt = LLAMA_HALLUCINATION_PROMPT_NO_DOC.format(
            question=question,
            answer=answer
        )

    response = ollama_client.chat(
        model=LLAMA_DETECTION_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response['message']['content'].strip()

    # Try to parse JSON from response
    try:
        # Find JSON in response (it might have extra text)
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            parsed = json.loads(json_str)

            # Normalize score from various possible formats
            score = parsed.get('SCORE', 'UNKNOWN')
            if isinstance(score, bool):
                score = 'FAIL' if score else 'PASS'
            elif isinstance(score, str):
                score = score.upper()
                if score not in ('PASS', 'FAIL'):
                    # Check HALLUCINATION_DETECTED field as backup
                    if parsed.get('HALLUCINATION_DETECTED', False):
                        score = 'FAIL'
                    else:
                        score = 'PASS'

            return {
                'reasoning': parsed.get('REASONING', []),
                'score': score,
                'confidence': parsed.get('CONFIDENCE', 'MEDIUM'),
                'suspicious_claims': parsed.get('SUSPICIOUS_CLAIMS', []),
                'hallucination_detected': parsed.get('HALLUCINATION_DETECTED', score == 'FAIL'),
                'raw_response': raw_response
            }
    except json.JSONDecodeError:
        pass

    # Fallback: extract score from text using multiple indicators
    score = 'UNKNOWN'
    confidence = 'LOW'

    raw_upper = raw_response.upper()

    # Check for explicit FAIL/PASS
    if 'SCORE": "FAIL' in raw_upper or 'SCORE":"FAIL' in raw_upper:
        score = 'FAIL'
    elif 'SCORE": "PASS' in raw_upper or 'SCORE":"PASS' in raw_upper:
        score = 'PASS'
    # Check for hallucination detected indicators
    elif 'HALLUCINATION_DETECTED": TRUE' in raw_upper or 'HALLUCINATION DETECTED' in raw_upper:
        score = 'FAIL'
    elif 'HALLUCINATION_DETECTED": FALSE' in raw_upper or 'NO HALLUCINATION' in raw_upper:
        score = 'PASS'
    # Last resort: look for keywords
    elif 'FAIL' in raw_upper and 'PASS' not in raw_upper:
        score = 'FAIL'
    elif 'PASS' in raw_upper and 'FAIL' not in raw_upper:
        score = 'PASS'

    return {
        'reasoning': raw_response,
        'score': score,
        'confidence': confidence,
        'suspicious_claims': [],
        'hallucination_detected': score == 'FAIL',
        'raw_response': raw_response
    }


def detect_hallucination(question: str, answer: str, document: str = "", model: str = None) -> dict:
    """
    Detect if an answer contains hallucinations using the configured detection model.

    This is the main entry point for hallucination detection. It dispatches to either
    Patronus Lynx (specialized) or Llama 3.2 (general purpose) based on configuration.

    Args:
        question: The question that was asked
        answer: The answer to evaluate
        document: Reference document/context (optional)
        model: Detection model to use ("lynx" or "llama"). If None, uses DETECTION_MODEL config.

    Returns:
        dict with 'reasoning', 'score', 'confidence', and 'raw_response'
    """
    # Use configured model if not specified
    if model is None:
        model = DETECTION_MODEL

    model = model.lower()

    if model == "lynx":
        return detect_hallucination_lynx(question, answer, document)
    elif model == "llama":
        return detect_hallucination_llama(question, answer, document)
    else:
        raise ValueError(f"Unknown detection model: {model}. Use 'lynx' or 'llama'.")


def run_hallucination_detection(input_file: Path = None, limit: int = None, detection_model: str = None):
    """
    Run hallucination detection on question-answer pairs.

    Args:
        input_file: Path to parquet file with 'question' and 'answer' columns.
                   If None, uses the file based on current LLM_PROVIDER setting.
        limit: Optional limit on number of pairs to process (for testing)
        detection_model: Model to use for detection ("lynx" or "llama").
                        If None, uses DETECTION_MODEL from config.
    """
    # Use configured model if not specified
    if detection_model is None:
        detection_model = DETECTION_MODEL

    print(f"Using detection model: {detection_model}")
    if detection_model == "llama":
        print(f"  Llama model: {LLAMA_DETECTION_MODEL}")
    else:
        print(f"  Lynx model: {LYNX_MODEL}")
    # Determine input file if not specified
    if input_file is None:
        if LLM_PROVIDER == "anthropic":
            input_file = Path('../data/hallucinationQuestions_{ANTHROPIC_MODEL}.parquet')
        else:
            input_file = Path('../data/hallucinationQuestions_{OLLAMA_MODEL}.parquet')

    df = pd.read_parquet(input_file, engine='fastparquet')

    if limit:
        df = df[:limit]

    print(f"Loaded {len(df)} question-answer pairs from {input_file}")

    # Run hallucination detection on all Q&A pairs
    results = []
    for i, row in df.iterrows():
        result = detect_hallucination(
            question=row['question'],
            answer=row['answer'],
            model=detection_model
        )

        # Normalize reasoning to string (can be list from JSON or string from fallback)
        reasoning = result['reasoning']
        if isinstance(reasoning, list):
            reasoning = json.dumps(reasoning, indent=2)

        result_entry = {
            'question': row['question'],
            'answer': row['answer'],
            'hallucination_score': result['score'],
            'confidence': result.get('confidence', 'N/A'),
            'reasoning': reasoning,
            'raw_response': result['raw_response'],
            'detection_model': detection_model
        }

        # Add Llama-specific fields if available
        if detection_model == "llama":
            suspicious = result.get('suspicious_claims', [])
            # Convert list to JSON string for parquet compatibility
            if isinstance(suspicious, list):
                suspicious = json.dumps(suspicious)
            result_entry['suspicious_claims'] = suspicious
            result_entry['hallucination_detected'] = result.get('hallucination_detected', False)

        results.append(result_entry)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(df)} pairs")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save results with model name in filename
    output_file = Path(f'../data/hallucinationDetection_{detection_model}.parquet')
    df_results.to_parquet(output_file, engine='fastparquet', index=False)

    # Summary statistics
    pass_count = (df_results['hallucination_score'] == 'PASS').sum()
    fail_count = (df_results['hallucination_score'] == 'FAIL').sum()
    unknown_count = (df_results['hallucination_score'] == 'UNKNOWN').sum()

    print(f"\n=== Hallucination Detection Results ===")
    print(f"Total pairs evaluated: {len(df_results)}")
    print(f"PASS (faithful): {pass_count} ({100*pass_count/len(df_results):.1f}%)")
    print(f"FAIL (hallucination detected): {fail_count} ({100*fail_count/len(df_results):.1f}%)")
    print(f"UNKNOWN: {unknown_count} ({100*unknown_count/len(df_results):.1f}%)")
    print(f"\nSaved to: {output_file}")

    return df_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate hallucination questions and detect hallucinations")
    parser.add_argument("--target-count", "-n", type=int, default=TARGET_COUNT,
                        help=f"Number of hallucination questions to generate (default: {TARGET_COUNT})")
    parser.add_argument("--sample-size", "-s", type=int, default=SAMPLE_SIZE,
                        help=f"Number of base questions to sample from (default: {SAMPLE_SIZE})")
    parser.add_argument("--provider", "-p", choices=["ollama", "anthropic"], default=None,
                        help="LLM provider to use (default: from config)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip question generation and only run detection")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip hallucination detection")
    parser.add_argument("--detection-limit", "-l", type=int, default=None,
                        help="Limit number of Q&A pairs to process for detection (for testing)")
    parser.add_argument("--input-file", "-i", type=str, default=None,
                        help="Input parquet file for detection (overrides auto-selection)")
    parser.add_argument("--detection-model", "-d", choices=["lynx", "llama"], default=None,
                        help="Hallucination detection model: 'lynx' (Patronus Lynx, specialized) or 'llama' (Llama 3.2, general purpose). Default: from DETECTION_MODEL env/config")

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

    # Generate hallucination questions and answers
    if not args.skip_generation:
        output_file = generate_hallucination_questions(
            target_count=args.target_count,
            sample_size=args.sample_size
        )
    else:
        # Use input file or default based on provider
        if args.input_file:
            output_file = Path(args.input_file)
        elif LLM_PROVIDER == "anthropic":
            output_file = Path('../data/hallucinationQuestions_anthropic.parquet')
        else:
            output_file = Path('../data/hallucinationQuestions_gemma3.parquet')
        print(f"Skipping generation, using existing file: {output_file}")

    # Run hallucination detection
    if not args.skip_detection:
        run_hallucination_detection(
            input_file=output_file,
            limit=args.detection_limit,
            detection_model=args.detection_model
        )
