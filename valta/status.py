from valta.config import load_state


def project_status(settings) -> str:
    """Where work is written, and which personal model is active."""
    state = load_state(settings)
    personal_model = state.get("personal_model") or settings.self_model
    return f"Project: {settings.root}\nPersonal model: {personal_model}"
