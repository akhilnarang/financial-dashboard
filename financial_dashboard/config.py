"""Application settings and Fernet key factory for financial-dashboard.

Settings are loaded from environment variables or a .env file via
pydantic-settings. All fields have defaults except EMAIL_SOURCE_MASTER_KEY,
which should be set to a Fernet key generated with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If EMAIL_SOURCE_MASTER_KEY is not set, get_fernet() generates an ephemeral
key and emits a warning. Credentials encrypted with an ephemeral key will
not survive a server restart.

get_fernet() is a factory function (not a module-level instance) so that it
can be called lazily from functions that need it. This avoids import-time
failures when the key is not yet set in the environment.

ALLOW_EPHEMERAL_MASTER_KEY (default False): the startup guard
(services.settings.assert_master_key_or_no_secrets) refuses to boot when
EMAIL_SOURCE_MASTER_KEY is unset but encrypted data already exists in the DB,
because an ephemeral key would silently render those secrets undecryptable.
Set this to true to downgrade that fatal refusal to a warning (e.g. for a
throwaway/dev DB you don't mind re-entering secrets in).
"""

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    db_url: str = "sqlite+aiosqlite:///./data/financial_dashboard.db"

    email_source_master_key: str = ""  # Fernet key for encrypting credentials

    # When True, the startup guard downgrades a missing master key + existing
    # encrypted data from a fatal SystemExit to a warning. See module docstring.
    allow_ephemeral_master_key: bool = False

    # HTTP Basic Auth — both must be set to enable, or both unset to disable.
    auth_username: str = ""
    auth_password: SecretStr = SecretStr("")

    # comma-separated CIDRs to skip auth
    auth_skip_cidrs: str = ""

    @model_validator(mode="after")
    def _validate_auth_pair(self):
        has_user = bool(self.auth_username)
        has_pass = bool(self.auth_password.get_secret_value())
        if has_user != has_pass:
            raise ValueError(
                "AUTH_USERNAME and AUTH_PASSWORD must both be set or both be empty."
            )
        return self

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_username)


settings = Settings()


_fernet_instance = None


def get_fernet():
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance

    from cryptography.fernet import Fernet

    key = settings.email_source_master_key
    if not key:
        import warnings

        key = Fernet.generate_key().decode()
        warnings.warn(
            "No EMAIL_SOURCE_MASTER_KEY set. Generated ephemeral key. "
            "Set it in .env for persistence."
        )
    _fernet_instance = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet_instance
