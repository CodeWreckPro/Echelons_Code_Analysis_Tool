from typing import List, Dict
from transformers import pipeline
from app.models.evolution import ChangeMetrics
from app.services.embedding import EmbeddingService

class StorytellingService:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.embedding_service = EmbeddingService()
        
        # Load templates for different types of changes
        self.templates = {
            "feature": "This change introduces {feature} to {component}, affecting {impact}.",
            "bugfix": "This fixes {issue} in {component} by {solution}.",
            "refactor": "This refactors {component} by {changes}, improving {benefit}.",
            "performance": "This improves performance of {component} by {changes}, resulting in {benefit}.",
            "security": "This enhances security by {changes} in {component}, protecting against {threats}."
        }

    def generate_commit_story(
        self,
        commit_message: str,
        changes: List[ChangeMetrics]
    ) -> Dict:
        """Generate a human-readable story about a commit."""
        # Analyze commit type
        commit_type = self._determine_commit_type(commit_message)
        
        # Analyze affected components
        components = self._identify_affected_components(changes)
        
        # Generate impact analysis
        impact = self._analyze_system_impact(changes)
        
        # Generate purpose explanation
        purpose = self._generate_purpose_explanation(
            commit_type,
            commit_message,
            components
        )
        
        return {
            "purpose": purpose,
            "impact": impact,
            "components": components
        }

    def generate_pr_summary(self, pr_url: str) -> Dict:
        """Generate a summary of a pull request."""
        # This is a placeholder for PR analysis
        # In a real implementation, you would:
        # 1. Fetch PR data from GitHub/GitLab API
        # 2. Analyze all commits in the PR
        # 3. Generate a comprehensive summary
        return {
            "summary": "Pull request analysis not implemented yet",
            "impact": "Unknown",
            "risk_level": "Unknown"
        }

    def _determine_commit_type(self, commit_message: str) -> str:
        """Determine the type of change from the commit message."""
        message_lower = commit_message.lower()
        
        if any(word in message_lower for word in ["add", "feature", "implement"]):
            return "feature"
        elif any(word in message_lower for word in ["fix", "bug", "issue"]):
            return "bugfix"
        elif any(word in message_lower for word in ["refactor", "clean", "restructure"]):
            return "refactor"
        elif any(word in message_lower for word in ["optimize", "performance", "speed"]):
            return "performance"
        elif any(word in message_lower for word in ["security", "vulnerability", "secure"]):
            return "security"
        else:
            return "other"

    def _identify_affected_components(self, changes: List[ChangeMetrics]) -> List[str]:
        """Identify the main components affected by the changes."""
        components = set()
        
        for change in changes:
            # Extract component from file path
            path_parts = change.file_path.split('/')
            if len(path_parts) > 1:
                component = path_parts[1]  # Assuming component is the first directory
                components.add(component)
        
        return list(components)

    def _analyze_system_impact(self, changes: List[ChangeMetrics]) -> str:
        """Analyze the potential system impact of the changes."""
        total_lines_changed = sum(c.lines_added + c.lines_deleted for c in changes)
        total_complexity_delta = sum(c.complexity_after - c.complexity_before for c in changes)
        
        impact_level = self._calculate_impact_level(
            total_lines_changed,
            total_complexity_delta,
            len(changes)
        )
        
        return self._generate_impact_description(
            impact_level,
            changes
        )

    def _calculate_impact_level(
        self,
        lines_changed: int,
        complexity_delta: float,
        files_changed: int
    ) -> str:
        """Calculate the impact level based on change metrics."""
        if lines_changed > 500 or complexity_delta > 20 or files_changed > 10:
            return "high"
        elif lines_changed > 100 or complexity_delta > 5 or files_changed > 3:
            return "medium"
        else:
            return "low"

    def _generate_impact_description(
        self,
        impact_level: str,
        changes: List[ChangeMetrics]
    ) -> str:
        """Generate a description of the system impact."""
        if impact_level == "high":
            return (
                "This change has significant system-wide impact, affecting multiple "
                "components and requiring careful testing and deployment."
            )
        elif impact_level == "medium":
            return (
                "This change has moderate impact on the system, affecting some "
                "key components but with manageable risk."
            )
        else:
            return (
                "This change has minimal system impact, affecting isolated components "
                "with low risk of side effects."
            )

    def _generate_purpose_explanation(
        self,
        commit_type: str,
        commit_message: str,
        components: List[str]
    ) -> str:
        """Generate a clear explanation of the change's purpose."""
        # Get the appropriate template
        template = self.templates.get(commit_type, self.templates["feature"])
        
        # Extract key information from commit message
        summary = self.summarizer(
            commit_message,
            max_length=100,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        
        # Fill in template
        if commit_type == "feature":
            return template.format(
                feature=summary,
                component=", ".join(components),
                impact="the system's functionality"
            )
        elif commit_type == "bugfix":
            return template.format(
                issue=summary,
                component=", ".join(components),
                solution="implementing necessary corrections"
            )
        elif commit_type == "refactor":
            return template.format(
                component=", ".join(components),
                changes=summary,
                benefit="code maintainability and readability"
            )
        elif commit_type == "performance":
            return template.format(
                component=", ".join(components),
                changes=summary,
                benefit="better system performance"
            )
        elif commit_type == "security":
            return template.format(
                changes=summary,
                component=", ".join(components),
                threats="potential security vulnerabilities"
            )
        else:
            return summary