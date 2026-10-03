"""Versioned development-selected abstention thresholds and explicit incident context."""

from pydantic import BaseModel, ConfigDict, Field


class RoutingPolicy(BaseModel):
    """Store measured selective routing, without claiming calibrated probabilities."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_version: str
    dev_sha256: str
    min_score: float = Field(ge=0, le=1)
    min_margin: float = Field(ge=0, le=1)
    accepted: int = Field(ge=0)
    accepted_families: int = Field(ge=0)
    accepted_accuracy: float = Field(ge=0, le=1)
    family_mean_accuracy: float = Field(ge=0, le=1)
    total: int = Field(gt=0)
    selection_note: str


def load_policy(settings, model_version):
    """Ignore missing, corrupt or stale profiles while retaining conservative fixed defaults."""
    try:
        profile = RoutingPolicy.model_validate_json(
            settings.understanding_routing_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return profile if profile.model_version == model_version else None


def explicit_category(products, severity):
    """Describe an explicitly reported shared outage without inferring its root cause."""
    services = {p.product for p in products}
    if severity.rule != "reported_area_outage":
        return None
    if "broadband" in services and "mobile" not in services:
        return "broadband_outage"
    if "mobile" in services and not services & {"broadband", "home_wifi"}:
        return "mobile_coverage"
    return None


CATEGORY_PRODUCTS = {
    "broadband_outage": {"broadband", "router", "home_wifi"},
    "intermittent_broadband": {"broadband", "router", "home_wifi"},
    "slow_broadband": {"broadband", "router", "home_wifi"},
    "wifi_connectivity": {"broadband", "router", "home_wifi"},
    "router_ont_hardware": {"broadband", "router"},
    "billing_dispute": {"billing"},
    "payment_restoration": {"billing"},
    "iptv": {"iptv"},
    **{
        label: {"mobile"}
        for label in (
            "mobile_coverage",
            "mobile_data",
            "voice_call_failure",
            "number_porting",
            "sim_esim_activation",
            "sms_otp",
            "roaming",
        )
    },
}


def compatible_category(category, products):
    """Require a mentioned compatible service before accepting a model-only route."""
    return bool(CATEGORY_PRODUCTS.get(category, set()) & {p.product for p in products})
