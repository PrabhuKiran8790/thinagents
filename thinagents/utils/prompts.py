"""
This module contains the PromptConfig class, which is used to construct the prompt for the agent.
"""

from typing import List, Optional, Tuple, Union

# SectionType defines the structure for individual sections that can be added to the prompt.
# A section can be a 2-tuple (heading, content) or a 3-tuple (heading, content, extra_text).
SectionType = Union[
    Tuple[str, Union[str, List[str]]], Tuple[str, Union[str, List[str]], Optional[str]]
]


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
    _field_order: List[str]

    def __init__(self, base_prompt: str):
        """
        Initializes the PromptConfig with a base prompt.

        Args:
            base_prompt (str): The base prompt for the agent.
        """
        self.base_prompt = base_prompt
        self.instructions = None
        self.context = None
        self.sections = None
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

    def build(self) -> str:
        """
        Constructs the final prompt string based on the added components.

        The order of components (instructions, context, sections) in the output
        string is determined by the order in which their respective `with_` or `add_`
        methods were called. The base prompt always comes first.

        Returns:
            str: The fully constructed prompt string.
        """
        parts = [self.base_prompt.strip()]

        for field_key in self._field_order:
            if field_key == "instructions" and self.instructions:
                instruction_block = ["Instructions:"]
                instruction_block.extend(f"- {i}" for i in self.instructions)
                parts.append("\n".join(instruction_block))
            elif field_key == "context" and self.context:
                parts.append(f"Context:\n{self.context}")
            elif field_key == "sections" and self.sections:
                for heading, content, extra_text in self.sections:
                    section_lines = [f"{heading}:"]
                    if isinstance(content, list):
                        section_lines.extend(f"- {line}" for line in content)
                    else:
                        section_lines.append(str(content))  # Ensure content is string
                    if extra_text:
                        # Ensure extra_text is string and add a newline if it's not empty.
                        # The original code had f"\n{extra_text}", which is fine if extra_text
                        # is meant to be a paragraph. If it's a short note, it might
                        # depend on desired formatting. Keeping original logic.
                        section_lines.append(f"\n{extra_text}")
                    parts.append("\n".join(section_lines))

        # Join all parts with double newlines, filtering out empty or whitespace-only parts.
        return "\n\n".join(part.strip() for part in parts if part and part.strip())
