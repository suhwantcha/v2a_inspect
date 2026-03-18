from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import streamlit as st
import streamlit_authenticator as stauth
import yaml

from v2a_inspect.settings import settings


def require_authentication() -> Any:
    credentials_path = resolve_auth_credentials_path()
    auth_cookie_key = settings.auth_cookie_key
    if auth_cookie_key is None:
        st.error("AUTH_COOKIE_KEY가 설정되지 않았습니다.")
        st.stop()
        raise RuntimeError("AUTH_COOKIE_KEY가 필요합니다.")

    auth_cookie_key_value = auth_cookie_key.get_secret_value()

    if not credentials_path.exists():
        st.error(
            "인증 설정 파일을 찾을 수 없습니다. "
            f"AUTH_CREDENTIALS_PATH를 설정하거나 {credentials_path} 파일을 추가하세요."
        )
        st.stop()

    auth_config = yaml.safe_load(credentials_path.read_text(encoding="utf-8")) or {}
    credentials_value = auth_config.get("credentials")
    if not isinstance(credentials_value, dict) or not credentials_value:
        st.error(f"인증 설정 파일 {credentials_path} 에 'credentials' 항목이 없습니다.")
        st.stop()

    credentials = cast(dict[str, Any], credentials_value)

    authenticator = stauth.Authenticate(
        credentials,
        settings.auth_cookie_name,
        auth_cookie_key_value,
        settings.auth_cookie_expiry_days,
    )
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
    return (Path.cwd() / "credentials.yaml").resolve()
