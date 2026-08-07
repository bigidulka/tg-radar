import base64

import pytest

from tg_radar.bot.image_generator import AirbotImageGenerator
from tg_radar.config import Settings


class FakeResponse:
    def __init__(self, data=None, content=b"") -> None:
        self.status_code = 200
        self.text = ""
        self._data = data or {}
        self.content = content

    def json(self):
        return self._data

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self) -> None:
        self.posts = []

    async def post(self, url, headers=None, json=None, data=None, files=None):
        self.posts.append({"url": url, "headers": headers, "json": json, "data": data, "files": files})
        encoded = base64.b64encode(b"fake-png").decode("ascii")
        return FakeResponse({"data": [{"b64_json": encoded}]})

    async def get(self, url):
        return FakeResponse(content=b"fake-url-png")


class FakeAvatar:
    data = None


class FakeAvatarLoader:
    def __init__(self, data) -> None:
        self.data = data

    async def fetch_avatar(self, channel):
        avatar = FakeAvatar()
        avatar.data = self.data
        avatar.content_type = "image/png"
        return avatar


@pytest.mark.asyncio
async def test_image_generator_reuses_existing_file(tmp_path):
    existing = tmp_path / "02-demo.png"
    existing.write_bytes(b"old-png")
    client = FakeClient()
    generator = AirbotImageGenerator(Settings(bot_generated_images_path=str(tmp_path)), client=client)

    path = await generator.get_or_create({"channel": "demo", "air_elo": 1200})

    assert path == existing
    assert client.posts == []


@pytest.mark.asyncio
async def test_image_generator_creates_missing_file(tmp_path):
    client = FakeClient()
    generator = AirbotImageGenerator(
        Settings(
            bot_generated_images_path=str(tmp_path),
            llm_base_url="http://llm/v1",
            llm_api_key="key",
        ),
        client=client,
    )

    path = await generator.get_or_create(
        {
            "channel": "@demo",
            "air_elo": 2600,
            "metrics": {"fomo_pressure": {"score": 88}},
            "meme_profile": {"rank_name": "rank", "one_liner": "line"},
        }
    )

    assert path == tmp_path / "llmref-demo.png"
    assert path.read_bytes() == b"fake-png"
    assert (tmp_path / "generated-demo.json").is_file()
    assert client.posts[0]["url"] == "http://llm/v1/images/generations"
    assert client.posts[0]["json"]["model"] == "gpt-image-2"


@pytest.mark.asyncio
async def test_image_generator_uses_avatar_reference_for_missing_file(tmp_path):
    generator = AirbotImageGenerator(
        Settings(
            bot_generated_images_path=str(tmp_path),
            llm_base_url="http://llm/v1",
            llm_api_key="key",
        ),
        client=FakeClient(),
        avatar_loader=FakeAvatarLoader(_png_bytes((220, 30, 40))),
    )

    path = await generator.get_or_create({"channel": "demo", "air_elo": 1200})

    assert path == tmp_path / "llmref-demo.png"
    assert path.is_file()
    assert generator._client.posts[0]["url"] == "http://llm/v1/images/edits"
    assert generator._client.posts[0]["data"]["model"] == "gpt-image-2"
    assert generator._client.posts[0]["files"]["image"][0] == "demo-avatar.png"


@pytest.mark.asyncio
async def test_image_generator_ignores_legacy_file_when_avatar_reference_is_available(tmp_path):
    legacy = tmp_path / "02-demo.png"
    legacy.write_bytes(b"old-png")
    client = FakeClient()
    generator = AirbotImageGenerator(
        Settings(
            bot_generated_images_path=str(tmp_path),
            llm_base_url="http://llm/v1",
            llm_api_key="key",
        ),
        client=client,
        avatar_loader=FakeAvatarLoader(_png_bytes((220, 30, 40))),
    )

    path = await generator.get_or_create({"channel": "demo", "air_elo": 1200})

    assert path == tmp_path / "llmref-demo.png"
    assert path.read_bytes() == b"fake-png"
    assert client.posts[0]["url"] == "http://llm/v1/images/edits"


def _png_bytes(color):
    from io import BytesIO

    from PIL import Image

    out = BytesIO()
    Image.new("RGB", (512, 512), color).save(out, format="PNG")
    return out.getvalue()
