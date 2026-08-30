"""
guardrails.py - Input and Output Guardrail System for IRIS Fine-Tuned LLM Pipeline
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Configure Governance Logging
logging.basicConfig(
    filename="guardrail_audit.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("LLM_Governance")

VALID_SPECIES = ["setosa", "versicolor", "virginica"]

class InputGuardrail:
    """Task 3: Intercepts adversarial prompts and structural schema violations."""

    def __init__(self):
        # Rule-based blocklist for direct & indirect prompt injection
        self.injection_patterns = [
            r"(?i)\b(ignore previous|system override|new task:)\b",
            r"(?i)\b(you are now a|act as a|roleplay as)\b",
            r"(?i)\b(write a poem|what is 2\+2)\b"
        ]

    def validate_structure(self, input_text: str, version: str) -> Tuple[bool, str]:
        """Validates input against expected IRIS feature schemas."""
        if version == "v1":
            # Expects numerical key-value structure
            pattern = r"^sepal_length:\s*[\d\.]+,\s*sepal_width:\s*[\d\.]+,\s*petal_length:\s*[\d\.]+,\s*petal_width:\s*[\d\.]+"
            if not re.search(pattern, input_text.strip()):
                return False, "Input violates v1 raw feature schema format."
        elif version == "v2":
            # Expects natural language description template
            pattern = r"A flower specimen has a sepal length of [\d\.]+ cm, sepal width of [\d\.]+ cm, petal length of [\d\.]+ cm, and petal width of [\d\.]+ cm\. Identify the iris species\."
            if not re.search(pattern, input_text.strip()):
                return False, "Input violates v2 description schema format."
        return True, ""

    def inspect(self, input_text: str, version: str) -> Dict[str, Any]:
        # Check 1: Pattern/Intent-based Prompt Injection Detection
        for pattern in self.injection_patterns:
            if re.search(pattern, input_text):
                reason = f"Prompt Injection pattern detected matching rule: '{pattern}'"
                logger.warning(json.dumps({"event": "INPUT_BLOCKED", "reason": reason, "raw_input": input_text, "timestamp": datetime.utcnow().isoformat()}))
                return {"blocked": True, "reason": reason, "stage": "input_injection_check"}

        # Check 2: Structural Schema Validation
        valid_struct, reason = self.validate_structure(input_text, version)
        if not valid_struct:
            logger.warning(json.dumps({"event": "INPUT_BLOCKED", "reason": reason, "raw_input": input_text, "timestamp": datetime.utcnow().isoformat()}))
            return {"blocked": True, "reason": reason, "stage": "input_schema_check"}

        return {"blocked": False}

class OutputGuardrail:
    """Task 4: Scans model output for context leakage and format non-compliance."""

    def __init__(self):
        # Leakage indicators (fragments of system prompts, chat schemas, training examples)
        self.leakage_patterns = [
            r"(?i)\b(system prompt|context window|instructions given|training dataset|training examples)\b",
            r"(?i)\b(formatting constraints|contents of your)\b"
        ]

    def extract_and_validate_format(self, raw_output: str, version: str) -> Optional[str]:
        cleaned = raw_output.strip()
        if version == "v1":
            # Exact match check for v1
            label = cleaned.strip('"').strip(".").lower()
            if label in VALID_SPECIES:
                return label
        elif version == "v2":
            # Strict regex check for "This is Iris <species>."
            match = re.search(r"^This is Iris (setosa|versicolor|virginica)\.$", cleaned, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    def inspect(self, raw_output: str, version: str) -> Dict[str, Any]:
        # Check 1: Sensitive Context Leakage Scanning
        for pattern in self.leakage_patterns:
            if re.search(pattern, raw_output):
                reason = "Response contains system prompt or context leakage fragments."
                logger.warning(json.dumps({"event": "OUTPUT_FILTERED", "reason": reason, "raw_output": raw_output, "timestamp": datetime.utcnow().isoformat()}))
                return {
                    "blocked": True,
                    "reason": reason,
                    "fallback_response": "Block Notice: Output filtered due to sensitive context leakage.",
                }

        # Check 2: Output Format Violation Scanning
        parsed_species = self.extract_and_validate_format(raw_output, version)
        if not parsed_species:
            reason = f"Response failed format compliance check for model {version}."
            logger.warning(json.dumps({"event": "OUTPUT_FILTERED", "reason": reason, "raw_output": raw_output, "timestamp": datetime.utcnow().isoformat()}))
            return {
                "blocked": True,
                "reason": reason,
                "fallback_response": "Block Notice: Output format non-compliant.",
            }

        return {"blocked": False, "parsed_output": parsed_species, "safe_response": raw_output}

class GuardedPipeline:
    """Wraps Vertex AI model prediction with input and output governance controls."""

    def __init__(self, endpoint_name: str, version: str, mock: bool = False):
        self.endpoint_name = endpoint_name
        self.version = version
        self.mock = mock
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()

    def predict(self, input_text: str) -> Dict[str, Any]:
        # Step 1: Input Guardrail Interception
        input_check = self.input_guardrail.inspect(input_text, self.version)
        if input_check["blocked"]:
            return {"status": "BLOCKED_BY_INPUT_GUARDRAIL", "reason": input_check["reason"], "response": None}

        # Step 2: Forward to Fine-Tuned Model Endpoint
        raw_model_response = self._call_model(input_text)

        # Step 3: Output Guardrail Filtering
        output_check = self.output_guardrail.inspect(raw_model_response, self.version)
        if output_check["blocked"]:
            return {
                "status": "FILTERED_BY_OUTPUT_GUARDRAIL",
                "reason": output_check["reason"],
                "raw_response": raw_model_response,
                "response": output_check["fallback_response"],
            }

        return {
            "status": "PASSED",
            "raw_response": raw_model_response,
            "parsed_species": output_check["parsed_output"],
            "response": output_check["safe_response"],
        }

    def _call_model(self, input_text: str, max_retries: int = 5) -> str:
        if self.mock:
            return "setosa" if self.version == "v1" else "This is Iris setosa."
        
        import vertexai
        from vertexai.generative_models import GenerativeModel
        from google.api_core.exceptions import ResourceExhausted
        import time

        model = GenerativeModel(self.endpoint_name)
        delay = 15

        for attempt in range(1, max_retries + 1):
            try:
                # Bulletproof pacing: Wait 15 seconds before every request (~4 requests/minute)
                time.sleep(15)
                response = model.generate_content(input_text)
                return response.text.strip()
            except ResourceExhausted:
                if attempt == max_retries:
                    raise
                print(f"  [429 quota hit] retrying in {delay}s (attempt {attempt}/{max_retries})...")
                time.sleep(delay)
                delay *= 2