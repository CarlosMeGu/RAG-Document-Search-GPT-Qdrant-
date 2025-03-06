import json
from typing import Any, Dict, Union


from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from document_retriever import DocumentRetriever
from qdrant_service import QdrantService


class MedicalDocumentAgent:
    """
    MedicalDocumentAgent class that uses a MedicalDocumentRetriever to fetch contexts,
    then processes them via an LLM to extract test names, parameters, etc.
    """

    def __init__(self,
                 retriever: DocumentRetriever,
                 llm_model: str = "gpt-4o-mini"):
        """
        Args:
            retriever: An instance of MedicalDocumentRetriever
            llm_model: The ID of the LLM model to use.
        """
        self.retriever = retriever
        # Initialize the language model to generate the JSON output
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)

    @staticmethod
    def _clean_json_response(json_response: Union[str, Any]) -> str:
        """Clean JSON response from LLM output (remove markdown, etc.)."""
        # If the response is an LLM object with .content
        if hasattr(json_response, "content"):
            json_str = json_response.content
        else:
            json_str = str(json_response)

        # Clean up the string if it contains markdown code blocks
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        return json_str

    @staticmethod
    def _parse_json_safely(json_str: str, default_value: Any = None) -> Any:
        """Safely parse JSON string with error handling."""
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return default_value if default_value is not None else {"error": str(e)}

    def obtain_tests(self, query_text: str, limit: int = 5) -> Dict[str, Any]:
        """
        Extract a list of medical test names from documents based on a `query_text`.
        Returns a dict with the raw LLM output in "json_output".
        """
        # Retrieve document contexts
        contexts = self.retriever.retrieve_contexts(query_text, limit)
        context_str = "\n\n".join(contexts)

        # Build the prompt instructing the LLM to generate a JSON list of test names
        prompt = (
            "Extract ONLY the names of specific medical tests explicitly mentioned in the document. "
            "Do NOT infer or guess; only return tests that appear verbatim in the document. "
            "Return the final answer as JSON in this format:\n\n"
            '{"medical_tests": ["Test1", "Test2", ...]}'
            "\n\nIf no tests are found, return {'medical_tests': []}.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Query:\n{query_text}\n\n"
            "JSON Output:"
        )

        # Call the LLM
        #TODO make a JSON Parser, create pydantic classes and parsers
        json_output = self.llm(prompt)
        return {"json_output": json_output}

    def obtain_test_parameters(self, test_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        For a given medical test name, retrieve all parameters/measurements in that test.
        Returns a dict with the raw LLM output in "test_parameters".
        """
        # Generate query for this test's parameters
        query_text = (
            f"For the test called '{test_name}', extract each parameter or measurement that is explicitly mentioned. "
            f"Include any measured user values, as well as lower and upper limits (reference range) and units, if stated."
        )

        # Retrieve contexts for the test
        contexts = self.retriever.retrieve_contexts(query_text, limit)
        context_str = "\n\n".join(contexts)

        # Build the prompt
        prompt = (
            f"You have the following context about the test '{test_name}':\n\n"
            f"{context_str}\n\n"
            "Extract each parameter from this test and return a JSON in this format:\n\n"
            "{\n"
            f'  "test_name": "{test_name}",\n'
            '  "parameters": [\n'
            '    {\n'
            '      "name": "ParameterName",\n'
            '      "user_value": <number or null>,\n'
            '      "lower_limit": <number or null>,\n'
            '      "upper_limit": <number or null>,\n'
            '      "units": "<string or null>"\n'
            '    },\n'
            '    ...\n'
            '  ]\n'
            "}\n\n"
            "Instructions:\n"
            " - Use null if any field is not explicitly stated.\n"
            " - For the measured value, if it appears as e.g. \"85 mg / dL,\" then user_value=85, units=\"mg / dL.\".\n"
            " - For reference ranges, if e.g. \"55-99 mg / dL\" is stated, then lower_limit=55, upper_limit=99, "
            "units=\"mg / dL.\".\n"
            " - Do not invent or infer fields that aren't in the text.\n\n"
            "JSON Output only:"
        )

        # Call the LLM
        response = self.llm(prompt)
        return {"test_parameters": response.content}

    def obtain_all_test_parameters(self, limit_per_test: int = 10) -> Dict[str, Any]:
        """
        For each medical test found in the entire document, retrieve all parameters/measurements.
        Returns a dictionary with all tests and their parameters.
        """
        # 1) First, get the comprehensive list of tests
        query_names = (
            "Extract a comprehensive list of all medical tests explicitly mentioned in the document "
            "as part of the medical exam. "
            "Ensure no test is omitted, and retrieve data from all pages. "
            "Do not include any tests that are not explicitly stated in the document."
        )

        # Reuse `obtain_tests` to get the JSON output
        result_names = self.obtain_tests(query_text=query_names, limit=30)

        # Parse the JSON from the response content
        try:
            json_str = self._clean_json_response(result_names["json_output"])
            tests_data = self._parse_json_safely(json_str, {"medical_tests": []})
            medical_tests = tests_data.get("medical_tests", [])
        except Exception as e:
            print(f"Error parsing medical tests JSON: {e}")
            return {"error": str(e)}

        # 2) For each test, get its parameters
        all_test_parameters = []
        for test_name in medical_tests:
            test_params = self.obtain_test_parameters(test_name, limit=limit_per_test)

            # Parse the parameters JSON
            try:
                params_json = self._clean_json_response(test_params["test_parameters"])
                test_data = self._parse_json_safely(params_json)
                all_test_parameters.append(test_data)
            except Exception as e:
                print(f"Error parsing parameters for {test_name}: {e}")
                all_test_parameters.append({
                    "test_name": test_name,
                    "parameters": [],
                    "error": str(e)
                })

        return {"all_tests": all_test_parameters}

if __name__ == "__main__":
    # 1) Initialize your QdrantService (ensure QDRANT_API_KEY and QDRANT_HOST are set)
    qdrant_service = QdrantService()

    # 2) Create the retriever
    retriever = DocumentRetriever(
        qdrant_service=qdrant_service,
        collection_name="documents",
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-4o-mini"
    )

    # 3) Create the agent
    agent = MedicalDocumentAgent(
        retriever=retriever,
        llm_model="gpt-4o-mini"
    )

    # 4) Run example queries to test functionality
    print("\n=== TEST: OBTAINING TEST NAMES ===")
    query_names = "Extract a list of all medical tests explicitly mentioned in the document."
    result_names = agent.obtain_tests(query_text=query_names, limit=30)
    print("result", result_names )
    print("Test Names JSON Output:\n", json.dumps(result_names, indent=2, ensure_ascii=False))

    print("\n=== TEST: OBTAINING PARAMETERS FOR A SPECIFIC TEST ===")
    test_name = "BIOMETRÍA HEMÁTICA"
    test_parameters = agent.obtain_test_parameters(test_name, limit=10)
    print(f"Parameters for {test_name}:\n", json.dumps(test_parameters, indent=2, ensure_ascii=False))

    print("\n=== TEST: OBTAINING ALL TEST PARAMETERS ===")
    all_test_parameters = agent.obtain_all_test_parameters(limit_per_test=10)
    print("All Test Parameters:\n", json.dumps(all_test_parameters, indent=2, ensure_ascii=False))

    print("\n=== TEST: ASKING A QUESTION BASED ON THE DOCUMENT ===")
    user_question = "What is the normal range for hemoglobin based on the document?"
    answer_response = retriever.answer_user_query(user_question, limit=10)
    print("User Question Answer:\n", answer_response["answer"])

    print("\n=== TEST: ASKING A QUESTION NOT RELATED TO THE DOCUMENT ===")
    unrelated_question = "Who won the FIFA World Cup in 2022?"
    unrelated_answer = retriever.answer_user_query(unrelated_question, limit=10)
    print("Unrelated Question Answer:\n", unrelated_answer["answer"])