from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from typing import Any

import streamlit as st
import streamlit_authenticator as stauth
import yaml

from v2a_inspect.settings import settings


@dataclass(frozen=True)
class DisabledAuthenticator:
    def logout(self, *_args: object, **_kwargs: object) -> None:
        return None


def require_authentication() -> Any:
    if settings.auth_mode == "disabled":
        _enable_guest_session()
        return DisabledAuthenticator()

    credentials_path = resolve_auth_credentials_path()
    ensure_auth_config_file(credentials_path)

    if not has_registered_users(credentials_path):
        if not settings.auth_allow_self_signup:
            st.error(
                "No accounts are configured yet and self-signup is disabled. "
                f"Create a credentials file at {credentials_path}."
            )
            st.stop()

        prepare_initial_signup(credentials_path)
        render_initial_signup(credentials_path)
        st.stop()

    authenticator = build_authenticator(credentials_path)
    authenticator.login()

    authentication_status = st.session_state.get("authentication_status")
    if authentication_status is False:
        st.error("Username 또는 Password가 올바르지 않습니다.")
        st.stop()
    if authentication_status is None:
        st.warning("Username과 Password를 입력해주세요.")
        st.stop()

    return authenticator


def resolve_auth_credentials_path() -> Path:
    if settings.auth_credentials_path is not None:
        return settings.auth_credentials_path.expanduser().resolve()

    appdata = os.environ.get("APPDATA")
    if appdata:
        return (Path(appdata) / "v2a_inspect" / "credentials.yaml").resolve()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return (Path(xdg_config_home) / "v2a_inspect" / "credentials.yaml").resolve()

    return (Path.home() / ".config" / "v2a_inspect" / "credentials.yaml").resolve()


def ensure_auth_config_file(credentials_path: Path) -> None:
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    if credentials_path.exists():
        config = load_auth_config(credentials_path)
    else:
        config = {}

    changed = False
    if not isinstance(config.get("credentials"), dict):
        config["credentials"] = {"usernames": {}}
        changed = True

    credentials = config["credentials"]
    if not isinstance(credentials.get("usernames"), dict):
        credentials["usernames"] = {}
        changed = True

    cookie = config.get("cookie")
    if not isinstance(cookie, dict):
        cookie = {}
        config["cookie"] = cookie
        changed = True

    cookie_name = settings.auth_cookie_name
    cookie_key = cookie.get("key") or resolve_auth_cookie_key()
    cookie_expiry_days = settings.auth_cookie_expiry_days
    if cookie.get("name") != cookie_name:
        cookie["name"] = cookie_name
        changed = True
    if cookie.get("key") != cookie_key:
        cookie["key"] = cookie_key
        changed = True
    if cookie.get("expiry_days") != cookie_expiry_days:
        cookie["expiry_days"] = cookie_expiry_days
        changed = True

    if changed or not credentials_path.exists():
        credentials_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def load_auth_config(credentials_path: Path) -> dict[str, Any]:
    if not credentials_path.exists():
        return {}

    loaded = yaml.safe_load(credentials_path.read_text(encoding="utf-8")) or {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def has_registered_users(credentials_path: Path) -> bool:
    config = load_auth_config(credentials_path)
    credentials = config.get("credentials")
    if not isinstance(credentials, dict):
        return False
    usernames = credentials.get("usernames")
    return isinstance(usernames, dict) and bool(usernames)


def prepare_initial_signup(credentials_path: Path) -> None:
    config = load_auth_config(credentials_path)
    pre_authorized = config.get("pre-authorized")
    if not isinstance(pre_authorized, dict):
        return

    emails = pre_authorized.get("emails")
    if isinstance(emails, list) and not emails:
        del config["pre-authorized"]
        credentials_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def build_authenticator(credentials_path: Path) -> stauth.Authenticate:
    return stauth.Authenticate(str(credentials_path))


def render_initial_signup(credentials_path: Path) -> None:
    st.title("Create First Account")
    st.info(
        "Authentication is enabled, but no users exist yet. "
        "Create the first account to finish setup."
    )
    st.caption(f"Credentials file: `{credentials_path}`")

    authenticator = build_authenticator(credentials_path)
    try:
        email, username, _name = authenticator.register_user(
            location="main",
            captcha=False,
            password_hint=False,
            clear_on_submit=True,
            key="Initial account setup",
            fields={
                "Form name": "Create first account",
                "First name": "First name",
                "Last name": "Last name",
                "Email": "Email",
                "Username": "Username",
                "Password": "Password",
                "Repeat password": "Repeat password",
                "Register": "Create account",
            },
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Account setup failed: {exc}")
        return

    if email and username:
        st.success(f"Created account `{username}`. Sign in with it now.")
        st.rerun()


def resolve_auth_cookie_key() -> str:
    if settings.auth_cookie_key is not None:
        return settings.auth_cookie_key.get_secret_value()
    return token_urlsafe(32)


def _enable_guest_session() -> None:
    st.session_state["authentication_status"] = True
    st.session_state["name"] = "Local User"
    st.session_state["username"] = "local-user"
