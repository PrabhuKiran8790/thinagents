"""
This module contains the PromptConfig class, which is used to construct the prompt for the agent.
"""

import re
from typing import List, Optional, Tuple, Union, Type
from pydantic import BaseModel, ValidationError


# SectionType defines the structure for individual sections that can be added to the prompt.
# A section can be a 2-tuple (heading, content) or a 3-tuple (heading, content, extra_text).
SectionType = Union[
    Tuple[str, Union[str, List[str]]], Tuple[str, Union[str, List[str]], Optional[str]]
]


class PromptingError(ValueError):
    """Custom exception for prompting-related errors, such as missing template variables."""
    pass


class PromptConfig:
    """
    Prompt constructor for the agent.

    This class allows for building a structured prompt by combining a base prompt
    with optional instructions, context, and custom sections. The order of these
    elements in the final prompt is determined by the order in which they are added.

    Attributes:
        base_prompt (str):
            The foundational text of the prompt.

        instructions (Optional[List[str]]):
            A list of instructions for the agent.
            Rendered as a bulleted list under an "Instructions:" heading.

        context (Optional[str]):
            Contextual information for the agent.
            Rendered under a "Context:" heading.

        sections (Optional[List[Tuple[str, Union[str, List[str]], Optional[str]]]]):
            A list of custom sections. Each section is a tuple containing:
            - heading (str): The title of the section.
            - content (Union[str, List[str]]): The main content of the section.
              If a list of strings, it's rendered as a bulleted list.
            - extra_text (Optional[str]): Additional text to append after the content.

        _field_order (List[str]):
            An internal list that tracks the order in which prompt components
            (instructions, context, sections) are added, ensuring they are built
            into the final prompt in that order.
    """

    base_prompt: str
    instructions: Optional[List[str]]
    context: Optional[str]
    sections: Optional[List[Tuple[str, Union[str, List[str]], Optional[str]]]]
    vars_schema: Optional[Type[BaseModel]]
    _field_order: List[str]

    def __init__(self, base_prompt: str, *, vars_schema: Optional[Type[BaseModel]] = None):
        """
        Initializes the PromptConfig with a base prompt and an optional variable schema.

        Args:
            base_prompt (str): The base prompt for the agent.
            vars_schema (Optional[Type[BaseModel]]): An optional Pydantic BaseModel class
                to validate the variables passed to the `build` method.
        """
        if vars_schema and not issubclass(vars_schema, BaseModel):
            raise TypeError("vars_schema must be a Pydantic BaseModel subclass.")

        self.base_prompt = base_prompt
        self.instructions = None
        self.context = None
        self.sections = None
        self.vars_schema = vars_schema
        self._field_order = []

    def _track_order(self, field: str):
        """
        Tracks the order of addition for different prompt parts.
        Ensures that parts are added to the prompt in the order they were set.
        """
        if field not in self._field_order:
            self._field_order.append(field)

    def with_instructions(self, instructions: List[str]):
        """
        Sets the instructions for the prompt. Replaces any existing instructions.

        Usage:
        ```python
        prompt = PromptConfig(base_prompt="You are a helpful assistant.")
        prompt.with_instructions(["Follow these guidelines.", "Be concise."])
        ```

        Args:
            instructions (List[str]): A list of instruction strings.

        Returns:
            PromptConfig: The instance of the PromptConfig for method chaining.
        """
        self.instructions = instructions
        self._track_order("instructions")
        return self

    def with_context(self, context: str):
        """
        Sets the context for the prompt. Replaces any existing context.

        Usage:
        ```python
        prompt = PromptConfig(base_prompt="You are a helpful assistant.")
        prompt.with_context("The user is asking about weather.")
        ```

        Args:
            context (str): A string containing contextual information.

        Returns:
            PromptConfig: The instance of the PromptConfig for method chaining.
        """
        self.context = context
        self._track_order("context")
        return self

    def with_sections(self, sections: List[SectionType]):
        """
        Sets custom sections for the prompt. Replaces any existing custom sections.

        Each section can be a 2-tuple (heading, content) or a
        3-tuple (heading, content, extra_text).

        Usage:
        ```python
        prompt = PromptConfig(base_prompt="You are a helpful assistant.")
        prompt.with_sections([
            ("User Query", "What's the capital of France?", "User is a student."),
            ("History", ["Previously asked about Spain."]) # No extra_text here
        ])
        ```

        Args:
            sections (List[SectionType]): A list of section tuples.

        Returns:
            PromptConfig: The instance of the PromptConfig for method chaining.

        Raises:
            ValueError: If any item in `sections` is not a 2-tuple or 3-tuple.
        """
        normalized: List[Tuple[str, Union[str, List[str]], Optional[str]]] = []
        for sec in sections:
            if len(sec) == 2:
                heading, content = sec
                normalized.append((heading, content, None))
            elif len(sec) == 3:
                # Type checker might not know sec is a 3-tuple here, but logic is sound.
                # We cast to ensure type consistency for `normalized` list.
                heading, content, extra = sec  # type: ignore
                normalized.append((heading, content, extra))
            else:
                raise ValueError(
                    "Section must be a 2-tuple (heading, content) or "
                    "3-tuple (heading, content, extra_text)."
                )
        self.sections = normalized
        self._track_order("sections")
        return self

    def add_instruction(self, instruction: str):
        """
        Adds a single instruction to the list of instructions.

        If no instructions exist, it initializes the list.

        Usage:
        ```python
        prompt = PromptConfig(base_prompt="You are a helpful assistant.")
        prompt.add_instruction("Be polite.")
        prompt.add_instruction("Answer truthfully.")
        ```

        Args:
            instruction (str): The instruction string to add.

        Returns:
            PromptConfig: The instance of the PromptConfig for method chaining.
        """
        if self.instructions is None:
            self.instructions = []
        self.instructions.append(instruction)
        self._track_order("instructions")
        return self

    def add_section(
        self,
        heading: str,
        content: Union[str, List[str]],
        extra_text: Optional[str] = None,
    ):
        """
        Adds a single custom section to the list of sections.

        If no sections exist, it initializes the list.

        Usage:
        ```python
        prompt = PromptConfig(base_prompt="You are a helpful assistant.")
        prompt.add_section("Critical Data", "Value: 42", "Handle with care.")
        prompt.add_section("Notes", ["Note 1", "Note 2"])
        ```

        Args:
            heading (str): The heading for the section.
            content (Union[str, List[str]]): The content for the section.
            extra_text (Optional[str], optional): Extra text for the section. Defaults to None.

        Returns:
            PromptConfig: The instance of the PromptConfig for method chaining.
        """
        if self.sections is None:
            self.sections = []
        self.sections.append((heading, content, extra_text))
        self._track_order("sections")
        return self

    def build(self, **kwargs) -> str:
        """
        Constructs the final prompt string based on the added components, with template substitution.

        This method first validates that all required template variables (e.g., `{name}`)
        are provided in the keyword arguments. If a `vars_schema` was provided during
        initialization, it will be used for robust type and structure validation.
        If any validation fails, it raises a `PromptingError`.

        The order of components (instructions, context, sections) in the output
        string is determined by the order in which their respective `with_` or `add_`
        methods were called. The base prompt always comes first.
        
        Example:
            prompt = PromptConfig("Hello, {name}.")
            prompt.build(name="World")  # Returns "Hello, World."

        Args:
            **kwargs: Keyword arguments for template substitution.

        Returns:
            str: The fully constructed prompt string.
            
        Raises:
            PromptingError: If any required template variables are missing or fail schema validation.
        """
        # --- 1. Pydantic Schema Validation (if provided) ---
        if self.vars_schema:
            try:
                self.vars_schema.model_validate(kwargs)
            except ValidationError as e:
                raise PromptingError(
                    f"Prompt variables failed validation for schema '{self.vars_schema.__name__}':\n{e}"
                ) from e

        # --- 2. Validate that all required template variables are present ---
        all_templates = [self.base_prompt]
        if self.instructions:
            all_templates.extend(self.instructions)
        if self.context:
            all_templates.append(self.context)
        if self.sections:
            for heading, content, extra_text in self.sections:
                all_templates.append(heading)
                if isinstance(content, list):
                    all_templates.extend(content)
                else:
                    all_templates.append(str(content))
                if extra_text:
                    all_templates.append(extra_text)

        required_vars = set()
        for template in all_templates:
            # Use regex to find all {variable} style placeholders
            required_vars.update(re.findall(r'{(\w+)}', template))

        provided_vars = set(kwargs.keys())
        missing_vars = required_vars - provided_vars

        if missing_vars:
            raise PromptingError(
                f"Missing required prompt variables: {sorted(list(missing_vars))}. "
                "Please provide them in the `prompt_vars` argument of agent.run()."
            )

        # --- 3. Build the prompt string ---
        parts = [self.base_prompt.strip().format(**kwargs)]

        for field_key in self._field_order:
            if field_key == "instructions" and self.instructions:
                formatted_instructions = [i.format(**kwargs) for i in self.instructions]
                instruction_block = ["Instructions:"]
                instruction_block.extend(f"- {i}" for i in formatted_instructions)
                parts.append("\n".join(instruction_block))
            elif field_key == "context" and self.context:
                formatted_context = self.context.format(**kwargs)
                parts.append(f"Context:\n{formatted_context}")
            elif field_key == "sections" and self.sections:
                for heading, content, extra_text in self.sections:
                    formatted_heading = heading.format(**kwargs)
                    section_lines = [f"{formatted_heading}:"]
                    if isinstance(content, list):
                        formatted_content_list = [line.format(**kwargs) for line in content]
                        section_lines.extend(f"- {line}" for line in formatted_content_list)
                    else:
                        formatted_content_str = str(content).format(**kwargs)
                        section_lines.append(formatted_content_str)
                    if extra_text:
                        formatted_extra_text = extra_text.format(**kwargs)
                        section_lines.append(f"\n{formatted_extra_text}")
                    parts.append("\n".join(section_lines))

        # Join all parts with double newlines, filtering out empty or whitespace-only parts.
        return "\n\n".join(part.strip() for part in parts if part and part.strip())
