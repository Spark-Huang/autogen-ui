from .datamodel import AgentConfig, ModelConfig, ToolConfig, TerminationConfig, TeamConfig
from autogen_agentchat.agents import AssistantAgent # AssistantAgent is already imported
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions._terminations import MaxMessageTermination, StopMessageTermination, TextMentionTermination
from autogen_core.tools._function_tool import FunctionTool


AgentTypes = AssistantAgent
TeamTypes = RoundRobinGroupChat | SelectorGroupChat
ModelTypes = OpenAIChatCompletionClient | None
TerminationTypes = MaxMessageTermination | StopMessageTermination | TextMentionTermination


# Define the custom speaker selector function (Corrected signature)
# It only receives the message history
import logging # Add logging for debugging

# Define the custom speaker selector function (Corrected signature)
# It only receives the message history
def custom_speaker_selector(messages: list) -> str | None:
    """
    Custom speaker selection based on message history:
    - Finds the last valid agent message (qa_agent or writing_agent) to base decisions on.
    - qa_agent speaks first and answers questions. If needed, it can request writing_agent
      by including 'REQUEST_WRITING_AGENT' in its message. It signals completion by
      including 'QA_COMPLETE'.
    - writing_agent speaks ONLY when requested by qa_agent. It signals completion by
      including 'STORY_COMPLETE'.
    - final_responder speaks once after either 'QA_COMPLETE' or 'STORY_COMPLETE'
      is found in the last valid agent message (case-insensitive).
    - The chat ends after final_responder speaks.
    """
    logging.debug(f"Selector called with {len(messages)} messages.")
    if not messages:
        logging.debug("Selector: No messages, selecting initial speaker: qa_agent")
        return "qa_agent"

    # --- Termination Check: Check if final_responder spoke anywhere ---
    logging.debug("Selector: --- Starting Termination Check ---") # Add start marker
    for idx, msg in enumerate(messages): # Add index for clarity
        speaker_name = None
        logging.debug(f"Selector: Checking message at index {idx}. Type: {type(msg)}, Content snippet: {str(msg)[:150]}...") # Log type and snippet

        # CORRECTED LOGIC: Prioritize 'source' attribute for TextMessage objects
        if hasattr(msg, 'source'):
            speaker_name = getattr(msg, 'source', None)
            logging.debug(f"Selector: Index {idx}: Found 'source' attribute: '{speaker_name}'")
        # Fallback for dict-like messages
        elif isinstance(msg, dict):
            speaker_name = msg.get('source', msg.get('name')) # Prefer 'source', fallback to 'name'
            logging.debug(f"Selector: Index {idx}: Message is dict. Extracted source/name: '{speaker_name}'")
        else:
            logging.debug(f"Selector: Index {idx}: Message is not object with 'source' or dict. Skipping name check.")

        # Check if the speaker is final_responder (case-insensitive)
        if speaker_name and isinstance(speaker_name, str):
            processed_name = speaker_name.strip().lower()
            logging.debug(f"Selector: Index {idx}: Comparing processed name '{processed_name}' with 'final_responder'")
            if processed_name == "final_responder":
                logging.debug(f"Selector: MATCH FOUND! final_responder (name from source/dict: '{speaker_name}') found at index {idx}. Terminating.")
                return None # Terminate the conversation
        else:
             logging.debug(f"Selector: Index {idx}: No valid speaker name found or not string. Speaker name was: '{speaker_name}'")

    logging.debug("Selector: --- Finished Termination Check (No termination) ---") # Add end marker

    # --- Find the last valid agent message ---
    last_valid_agent_message = None
    logging.debug("Selector: Searching for last valid agent message...")
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        logging.debug(f"Selector: Checking message at index {i}. Type: {type(msg)}")

        # Try accessing keys directly or via .get() if available
        source_raw = None
        content_raw = None
        try:
            # Prefer .get() if it's dict-like, otherwise try direct access if it's an object
            if hasattr(msg, 'get'):
                 source_raw = msg.get("source")
                 content_raw = msg.get("content")
            elif hasattr(msg, 'source') and hasattr(msg, 'content'):
                 source_raw = msg.source
                 content_raw = msg.content
        except Exception as e:
            logging.debug(f"Selector: Error accessing source/content at index {i}: {e}")
            continue # Skip if keys/attributes don't exist

        # Check if both source and content were successfully retrieved
        if source_raw is not None and content_raw is not None:
            source_type = type(source_raw)
            logging.debug(f"Selector: Index {i}: Found 'source' key/attr. Type: {source_type}, Raw value: '{source_raw}'")

            # Ensure source is string before processing
            if isinstance(source_raw, str):
                source_stripped = source_raw.strip()
                source_lower = source_stripped.lower()
                logging.debug(f"Selector: Index {i}: Stripped source: '{source_stripped}', Lowercased source: '{source_lower}'")

                # Check if source contains agent names (case-insensitive)
                is_qa_agent = "qa_agent" in source_lower
                is_writing_agent = "writing_agent" in source_lower
                logging.debug(f"Selector: Index {i}: 'qa_agent' in source? {is_qa_agent}, 'writing_agent' in source? {is_writing_agent}")

                if is_qa_agent or is_writing_agent:
                    last_valid_agent_message = msg # Store the original msg object
                    logging.debug(f"Selector: Found last valid agent message at index {i} from source '{source_lower}' (original source: '{source_raw}')")
                    break # Found the most recent valid one
                else:
                    logging.debug(f"Selector: Processed message source '{source_lower}' does not contain 'qa_agent' or 'writing_agent'. Skipping.")
            else:
                logging.debug(f"Selector: Message source at index {i} is not a string (Type: {source_type}). Skipping.")
        else:
            logging.debug(f"Selector: Message at index {i} lacks 'source' or 'content' key/attribute. Skipping. Msg: {msg}")


    if last_valid_agent_message is None:
        logging.warning("Selector: No valid agent message (containing 'qa_agent' or 'writing_agent' in source) found in history. Defaulting to qa_agent.")
        return "qa_agent" # Default if no valid agent message found

    # --- Speaker Selection based on LAST VALID AGENT message ---
    # Get source and content from the identified valid message (using the same safe access)
    last_valid_source_raw = None
    last_valid_content_raw = None
    try:
        if hasattr(last_valid_agent_message, 'get'):
             last_valid_source_raw = last_valid_agent_message.get("source")
             last_valid_content_raw = last_valid_agent_message.get("content")
        elif hasattr(last_valid_agent_message, 'source') and hasattr(last_valid_agent_message, 'content'):
             last_valid_source_raw = last_valid_agent_message.source
             last_valid_content_raw = last_valid_agent_message.content
    except Exception as e:
         logging.error(f"Selector: Error accessing source/content from identified last_valid_agent_message: {e}. Defaulting to qa_agent.")
         return "qa_agent" # Should not happen if found previously, but safety check

    last_valid_source = str(last_valid_source_raw).strip().lower() if isinstance(last_valid_source_raw, str) else "" # Processed source
    last_valid_content = str(last_valid_content_raw).strip().lower() if isinstance(last_valid_content_raw, str) else ""

    # 1. Trigger writing_agent if requested by qa_agent in the last valid message
    # Use processed source and content
    if "qa_agent" in last_valid_source and "request_writing_agent" in last_valid_content:
        logging.debug("Selector: Selecting writing_agent based on request in last valid message.")
        return "writing_agent"

    # 2. Trigger final_responder if a completion signal is in the last valid message
    # Use processed source and content
    if ("qa_agent" in last_valid_source and "qa_complete" in last_valid_content) or \
       ("writing_agent" in last_valid_source and "story_complete" in last_valid_content):
        logging.debug(f"Selector: Selecting final_responder based on completion signal in last valid message: '{last_valid_content}' from source '{last_valid_source_raw}'") # Log original source for clarity
        return "final_responder"

    # 3. Default: qa_agent
    logging.debug("Selector: No specific trigger in last valid message. Defaulting to qa_agent.")
    return "qa_agent"


class Provider():
    def __init__(self):
        pass
    def load_model(self, model_config: ModelConfig | dict) -> ModelTypes:
        if isinstance(model_config, dict):
            try:
                model_config = ModelConfig(**model_config)
            except:
                raise ValueError("Invalid model config")
        model = None
        if model_config.model_type == "OpenAIChatCompletionClient":
            # Prepare arguments for the client constructor
            client_args = {
                "model": model_config.model,
            }
            # Add optional arguments if they exist in the config
            if hasattr(model_config, 'base_url') and model_config.base_url:
                client_args['base_url'] = model_config.base_url
            # Crucially, pass model_info if it exists, as required for non-standard model names
            if hasattr(model_config, 'model_info') and model_config.model_info:
                # Assuming model_info in config is already a dict or compatible structure
                client_args['model_info'] = model_config.model_info

            # Instantiate the client using the prepared arguments
            # The client typically handles API key from environment variables (e.g., OPENAI_API_KEY)
            try:
                 model = OpenAIChatCompletionClient(**client_args)
            except ValueError as e:
                 # Provide more context if the specific error occurs again
                 print(f"Error initializing OpenAIChatCompletionClient: {e}")
                 print(f"Arguments passed: {client_args}")
                 raise e # Re-raise the original error
            except Exception as e:
                 print(f"An unexpected error occurred during client initialization: {e}")
                 raise e
        return model

    def _func_from_string(self, content: str) -> callable:
        """
        Convert a string containing function code into a callable function object.

        Args:
            content (str): String containing the function code, with proper indentation

        Returns:
            Callable: The compiled function object
        """
        # Create a namespace for the function
        namespace = {}

        # Ensure content is properly dedented if it contains indentation
        lines = content.split('\n')
        if len(lines) > 1:
            # Find the minimum indentation (excluding empty lines)
            indents = [len(line) - len(line.lstrip())
                       for line in lines if line.strip()]
            min_indent = min(indents) if indents else 0
            # Remove the minimum indentation from each line
            lines = [line[min_indent:]
                     if line.strip() else line for line in lines]
            content = '\n'.join(lines)

        try:
            # Execute the function definition in the namespace
            exec(content, namespace)

            # Find and return the function object
            # Get the first callable object from the namespace
            for item in namespace.values():
                if callable(item) and not isinstance(item, type):
                    return item

            raise ValueError("No function found in the provided code")
        except Exception as e:
            raise ValueError(
                f"Failed to create function from string: {str(e)}")

    def load_tool(self, tool_config: ToolConfig | dict) -> FunctionTool:
        if isinstance(tool_config, dict):
            try:
                tool_config = ToolConfig(**tool_config)
            except:
                raise ValueError("Invalid tool config")
        tool = FunctionTool(name=tool_config.name, description=tool_config.description,
                            func=self._func_from_string(tool_config.content))
        return tool

    def load_agent(self, agent_config: AgentConfig | dict) -> AgentTypes:
        if isinstance(agent_config, dict):
            try:
                agent_config = AgentConfig(**agent_config)
            except:
                raise ValueError("Invalid agent config")
        agent = None
        if agent_config.agent_type == "AssistantAgent":
            model_client = self.load_model(agent_config.model_client)
            system_message = agent_config.system_message if agent_config.system_message else "You are a helpful AI assistant. Solve tasks using your tools. Reply with 'TERMINATE' when the task has been completed."
            # Handle cases where tools might be None before iterating
            tools = []
            if agent_config.tools is not None:
                tools = [self.load_tool(tool) for tool in agent_config.tools]
            
            agent = AssistantAgent(
                name=agent_config.name, model_client=model_client, tools=tools, system_message=system_message)

        return agent

    def load_termination(self, termination_config: TerminationConfig | dict) -> TerminationTypes:
        if isinstance(termination_config, dict):
            try:
                termination_config = TerminationConfig(**termination_config)
            except:
                raise ValueError("Invalid termination config")
        termination = None
        if termination_config.termination_type == "MaxMessageTermination":
            termination = MaxMessageTermination(
                max_messages=termination_config.max_messages)
        elif termination_config.termination_type == "StopMessageTermination":
            termination = StopMessageTermination()
        elif termination_config.termination_type == "TextMentionTermination":
            termination = TextMentionTermination(text=termination_config.text)
        return termination

    def load_team(self, team_config: TeamConfig | dict) -> TeamTypes:
        if isinstance(team_config, dict):
            try:
                team_config = TeamConfig(**team_config)
            except:
                raise ValueError("Invalid team config")
        team = None
        agents = []
        termination = self.load_termination(team_config.termination_condition)
        # tbd on termination condition
        for agent_config in team_config.participants:
            agent = self.load_agent(agent_config)
            agents.append(agent)
        if team_config.team_type == "RoundRobinGroupChat":
            team = RoundRobinGroupChat(
                agents, termination_condition=termination)
        elif team_config.team_type == "SelectorGroupChat":
            # Load the top-level model client ONLY if it's defined in the config
            top_level_client = None
            if team_config.model_client:
                 top_level_client = self.load_model(team_config.model_client)
            else:
                 # Explicitly handle the case where no model_client is needed/provided
                 logging.debug("SelectorGroupChat: No top-level model_client configured. Relying solely on selector_func.")

            # Pass agents as the first positional argument
            # Use the correct 'selector_func' parameter name
            team = SelectorGroupChat(
                agents, # Positional argument
                termination_condition=termination,
                model_client=top_level_client, # Pass the potentially None client
                selector_func=custom_speaker_selector # Use correct parameter
            )

        return team
