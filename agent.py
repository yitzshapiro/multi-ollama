import requests
import logging
import random
import time
from fake_useragent import UserAgent
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from bs4 import BeautifulSoup
import json
import os
import dotenv
from playwright.sync_api import sync_playwright

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class PlanStep(BaseModel):
    step_desc: str
    command: str

class CommandOutput(BaseModel):
    PLAN: List[PlanStep]
    pageContextObjects: dict
    userInfo: dict
    status: Optional[str] = None

class PlaywrightAutomation:

    OLLAMA_BASE_URL = "http://localhost:11434/api"
    MODEL_NAME = "phi3:14b"
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    GOOGLE_SEARCH_CX = os.environ.get("GOOGLE_SEARCH_CX")

    user_profiles = {}
    session_data = {}

    def get_random_user_agent(self):
        ua = UserAgent()
        return ua.random

    def load_user_profile(self, user_id):
        profile_path = f"profiles/{user_id}.json"
        if os.path.exists(profile_path):
            with open(profile_path, "r") as f:
                return json.load(f)
        else:
            return {"user_id": user_id, "preferences": {}, "session_history": []}

    def save_user_profile(self, user_id, profile_data):
        os.makedirs("profiles", exist_ok=True)
        profile_path = f"profiles/{user_id}.json"
        with open(profile_path, "w") as f:
            json.dump(profile_data, f)

    def send_to_model(self, prompt):
        try:
            response = requests.post(
                f"{self.OLLAMA_BASE_URL}/generate",
                json={
                    "model": self.MODEL_NAME,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                },
            )

            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")

            print("Raw response from model:", raw_response)

            parsed_response = json.loads(raw_response)

            print("Parsed response:", json.dumps(parsed_response, indent=2))

            return CommandOutput(**parsed_response)

        except ValidationError as ve:
            logging.error(f"Validation Error: {ve}")
            print("Validation Error Details:")
            print(ve.json(indent=2))
            return None
        except Exception as e:
            logging.error(f"Error communicating with the model: {e}")
            print(f"Exception: {e}")
            return None

    def process_model_output(self, output):
        if output is None:
            return [], "", "NOT SURE"

        try:
            commands = [step.command for step in output.PLAN if step.command]
            explanation = " ".join(step.step_desc for step in output.PLAN)
            status = output.status if output.status else "NOT SURE"

            return commands, explanation, status
        except AttributeError as e:
            logging.error(f"Attribute error in model output processing: {e}")
            return [], "", "NOT SURE"

    def check_for_stagnation(self, previous_plan, current_plan):
        return previous_plan == current_plan

    def get_simplified_browser_content(self, page):
        page_source = page.content()
        soup = BeautifulSoup(page_source, "html.parser")

        for script in soup(["script", "style", "meta", "link", "noscript"]):
            script.decompose()

        visible_text = soup.get_text(separator=" ", strip=True)
        cleaned_text = " ".join(visible_text.split())

        return cleaned_text

    def get_all_interactable_elements(self, page):
        elements = {}
        try:

            input_fields = page.query_selector_all("input")
            for i, input_field in enumerate(input_fields):
                placeholder = input_field.get_attribute("placeholder")
                name = input_field.get_attribute("name")
                if placeholder:
                    elements[f"input_placeholder_{i}"] = (
                        f'input[placeholder="{placeholder}"]'
                    )
                elif name:
                    elements[f"input_name_{i}"] = f'input[name="{name}"]'
                else:
                    elements[f"input_{i}"] = f"input:nth-of-type({i + 1})"

            buttons = page.query_selector_all("button")
            for i, button in enumerate(buttons):
                aria_label = button.get_attribute("aria-label")
                text = button.inner_text().strip()
                if aria_label:
                    elements[f"button_aria_label_{i}"] = (
                        f'button[aria-label="{aria_label}"]'
                    )
                elif text:
                    elements[f"button_text_{i}"] = f'button:has-text("{text}")'
                else:
                    elements[f"button_{i}"] = f"button:nth-of-type({i + 1})"

            links = page.query_selector_all("a")
            for i, link in enumerate(links):
                href = link.get_attribute("href")
                text = link.inner_text().strip()
                if text:
                    elements[f"link_text_{i}"] = f'a:has-text("{text}")'
                elif href:
                    elements[f"link_href_{i}"] = f'a[href="{href}"]'
                else:
                    elements[f"link_{i}"] = f"a:nth-of-type({i + 1})"

            return elements

        except Exception as e:
            logging.error(f"Error getting all interactable elements: {e}")
            return {}

    def select_relevant_link(self, search_results, page):
        for i, result in enumerate(search_results):
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()

            if "vegan" in title or "restaurant" in snippet:
                logging.info(f"Found relevant search result: {title}")
                return f"link_text_{i}"

        logging.warning("No relevant search result found.")
        return None

    def execute_browser_commands(self, page, commands, elements):
        try:
            for command in commands:
                if not command:
                    logging.warning("Empty command found, skipping...")
                    continue

                parts = command.split()
                action = parts[0]

                if action == "GOTO_URL":
                    page.goto(parts[1])
                    logging.info(f"Navigated to {parts[1]}")
                elif action == "CLICK" and len(parts) > 1:
                    element_name = parts[1]
                    if element_name in elements:
                        page.click(elements[element_name])
                        logging.info(f"Clicked on {element_name}")
                    else:
                        logging.warning(f"No mapped element found for {element_name}")
                elif action == "TYPE" and len(parts) > 2:
                    element_name = parts[1]
                    text = " ".join(parts[2:])
                    if element_name in elements:
                        page.fill(elements[element_name], text)
                        logging.info(f"Typed '{text}' into {element_name}")
                    else:
                        logging.warning(f"No mapped element found for {element_name}")
                elif action == "SUBMIT" and len(parts) > 1:
                    element_name = parts[1]
                    if element_name in elements:
                        page.press(elements[element_name], "Enter")
                        logging.info(f"Submitted form via {element_name}")
                    else:
                        logging.warning(f"No mapped element found for {element_name}")
                elif action == "GOOGLE_SEARCH_API":
                    
                    logging.info(f"Executing Google Search API with query: {' '.join(parts[1:])}")
                    search_results = self.google_search(" ".join(parts[1:]))
                    logging.info("Search results retrieved via API")
                    
                    search_results_summary = "\n".join(
                        [f"Title: {item['title']}\nURL: {item['link']}" for item in search_results]
                    )
                    logging.info(f"Search results summary provided to LLM:\n{search_results_summary}")

                    return search_results  
                else:
                    logging.warning(f"Unknown command or insufficient parameters: {command}")
        except Exception as e:
            logging.error(f"Execution error: {e}")


    def manage_session(self, user_id, action, data=None):
        global session_data
        if action == "load":
            session_data = self.load_user_profile(user_id).get("session_history", [])
        elif action == "save":
            profile_data = self.load_user_profile(user_id)
            profile_data["session_history"].append(data)
            self.save_user_profile(user_id, profile_data)
        elif action == "clear":
            session_data = []

    def google_search(self, query):
        try:

            simplified_query = " ".join(query.split()[:10])

            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.GOOGLE_API_KEY,
                "cx": self.GOOGLE_SEARCH_CX,
                "q": simplified_query,
            }

            full_url = requests.Request("GET", url, params=params).prepare().url
            logging.info(f"Google Search API Request URL: {full_url}")

            response = requests.get(url, params=params)
            response.raise_for_status()

            search_results = response.json()
            logging.info("Search results fetched successfully.")
            return search_results.get("items", [])

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            logging.error(f"Response content: {http_err.response.content.decode()}")
        except Exception as e:
            logging.error(f"Error during Google search: {e}")
        return []

    def generate_final_output(self, model_output, search_results=None):
        """ Generate a final output summary based on the model's last response. """
        try:
            plan_summary = "\n".join([f"Step: {step.step_desc}" for step in model_output.PLAN])

            if search_results:
                results_summary = "\n".join(
                    [f"Title: {item['title']}\nURL: {item['link']}" for item in search_results]
                )
                final_summary = f"Task Summary:\n{plan_summary}\n\nKey URLs from Search Results:\n{results_summary}"
            else:
                final_summary = f"Task Summary:\n{plan_summary}\nNo URLs or specific results were found."

            return f"{final_summary}\nTask Status: {model_output.status}"
        except Exception as e:
            logging.error(f"Error generating final output: {e}")
            return "Final output could not be generated."

    def run_automation(self):
        user_id = "default_user"
        user_objective = input("Enter your objective: ")

        session_messages = ""  
        previous_plan = None
        current_plan = "No plan yet"
        model_output = None  
        search_results = None  

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            while True:
                self.manage_session(user_id, "load")

                
                system_prompt = f"""
                
                
                You are an expert agent named MULTI·ON developed by "MultiOn" controlling a browser (you are not just a language model anymore).
                You are given:
                1. An objective that you are trying to achieve: {user_objective}

                
                Start with navigation by using "GOTO_URL" followed by the URL.
                After navigation, inspect the page and issue the next commands based on the available elements.

                IMPORTANT:
                - Do not assume elements exist before seeing the page.
                - Use the available elements after loading the page.

                
                {current_plan}

                
                {session_messages}

                
                The output MUST be a JSON object following this structure:
                {{
                    "PLAN": [
                        {{
                            "step_desc": "Description of the step",
                            "command": "Command to be executed (e.g., GOTO_URL https://example.com)"
                        }}
                    ],
                    "pageContextObjects": {{}},
                    "userInfo": {{}},
                    "status": "DONE"  
                }}
                """

                
                model_output = self.send_to_model(system_prompt)

                
                if model_output is None or not model_output.PLAN:
                    logging.warning("No valid output from model, stopping execution.")
                    break

                
                commands, explanation, status = self.process_model_output(model_output)

                session_messages += f"\n{json.dumps(model_output.dict(), indent=2)}"

                print(f"Commands: {commands}")
                print(f"Status: {status}")

                current_plan = model_output.PLAN  

                if self.check_for_stagnation(previous_plan, current_plan):
                    logging.warning("Stagnation detected, exiting...")
                    break
                previous_plan = current_plan

                
                search_results = self.execute_browser_commands(page, [commands[0]], {})

                
                elements = self.get_all_interactable_elements(page)
                print(f"Elements found: {elements}")

                
                system_prompt = f"""
                
                
                You are an expert agent named MULTI·ON developed by "MultiOn" controlling a browser.
                You have successfully navigated to the page. The next steps involve interacting with the page elements.

                
                {list(elements.keys())}

                
                {current_plan}

                
                Based on the elements, choose appropriate actions like CLICK, TYPE, SUBMIT, etc.

                
                {session_messages}

                
                The output MUST be a JSON object following this structure:
                {{
                    "PLAN": [
                        {{
                            "step_desc": "Description of the step",
                            "command": "Command to be executed (e.g., CLICK input_placeholder_0)"
                        }}
                    ],
                    "pageContextObjects": {{}},
                    "userInfo": {{}},
                    "status": "DONE"  
                }}
                """

                
                model_output = self.send_to_model(system_prompt)

                if model_output is None or not model_output.PLAN:
                    logging.warning("No valid output from model after page load, stopping execution.")
                    break

                
                commands, explanation, status = self.process_model_output(model_output)
                session_messages += f"\n{json.dumps(model_output.dict(), indent=2)}"

                
                self.execute_browser_commands(page, commands, elements)

                
                self.manage_session(user_id, "save", session_messages)

                if status == "DONE":
                    logging.info("Task completed.")
                    final_output = self.generate_final_output(model_output, search_results)
                    print(f"Final Output: {final_output}")
                    break
                elif status == "NOT SURE" or status == "WRONG":
                    logging.warning("Model needs assistance.")
                    break
                else:
                    continue

            
            browser.close()

if __name__ == "__main__":
    PlaywrightAutomation().run_automation()
