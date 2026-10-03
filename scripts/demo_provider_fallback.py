"""Demonstrate provider failures with a synthetic complaint and unchanged credentials."""

import argparse
import sys

from app.config.settings import get_settings
from app.llm.client import LanguageUnavailable
from app.llm.gemini import GeminiClient
from app.llm.providers import ProviderChain
from app.resolution.models import ResolutionRequest
from app.resolution.service import ResolutionService
from app.understanding.service import UnderstandingService, get_understanding_service


class UnavailableProvider:
    """Simulate a provider outage without sending a request or changing configuration."""

    def __init__(self, name, model, error):
        self.name, self.model, self.error = name, model, error

    def generate(self, instruction, payload, schema):
        """Expose the same sanitized failure contract as real adapters."""
        raise LanguageUnavailable(self.error)


def main():
    """Use Gemini after a simulated Groq limit, or demonstrate the fully local fallback."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail", choices=["groq", "both"], default="both")
    args = parser.parse_args()
    settings = get_settings().model_copy(update={"llm_enabled": True})
    providers = [UnavailableProvider("groq", settings.groq_model, "provider_http_429")]
    providers.append(
        GeminiClient(settings)
        if args.fail == "groq"
        else UnavailableProvider("gemini", settings.gemini_model, "provider_http_503")
    )
    chain = ProviderChain(settings, providers=providers)
    understanding = UnderstandingService(
        get_understanding_service().classifier, settings=settings, language=chain
    )
    service = ResolutionService(understanding=understanding, settings=settings, language=chain)
    result = service.resolve(
        ResolutionRequest(
            query="My broadband drops. Ethernet works. All wireless devices disconnect. I already restarted the router."
        )
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
