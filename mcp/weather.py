#!/usr/bin/env python3
"""
Weather MCP Server
한국 주요 도시 날씨 조회 MCP 서버

Open-Meteo API를 사용하여 무료로 날씨 정보를 제공합니다.
"""

from mcp.server.fastmcp import FastMCP
import httpx
from datetime import datetime

# FastMCP 서버 초기화
mcp = FastMCP("weather-server")

# 주요 도시 좌표 (위도, 경도)
CITIES = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "인천": (37.4563, 126.7052),
    "대구": (35.8714, 128.6014),
    "대전": (36.3504, 127.3845),
    "광주": (35.1595, 126.8526),
    "울산": (35.5384, 129.3114),
    "제주": (33.4996, 126.5312),
}


def get_weather_description(code: int) -> str:
    """WMO 날씨 코드를 한글 설명으로 변환"""
    weather_codes = {
        0: "맑음",
        1: "대체로 맑음",
        2: "부분적으로 흐림",
        3: "흐림",
        45: "안개",
        48: "서리 안개",
        51: "가벼운 이슬비",
        53: "보통 이슬비",
        55: "강한 이슬비",
        61: "약한 비",
        63: "보통 비",
        65: "강한 비",
        71: "약한 눈",
        73: "보통 눈",
        75: "강한 눈",
        77: "진눈깨비",
        80: "약한 소나기",
        81: "보통 소나기",
        82: "강한 소나기",
        85: "약한 눈 소나기",
        86: "강한 눈 소나기",
        95: "뇌우",
        96: "약한 우박을 동반한 뇌우",
        99: "강한 우박을 동반한 뇌우",
    }
    return weather_codes.get(code, f"알 수 없는 날씨 (코드: {code})")


@mcp.tool()
async def get_weather(city: str = "서울") -> str:
    """
    특정 도시의 현재 날씨 정보를 가져옵니다.

    Args:
        city: 도시 이름 (서울, 부산, 인천, 대구, 대전, 광주, 울산, 제주 중 선택)

    Returns:
        날씨 정보 문자열
    """
    if city not in CITIES:
        available_cities = ", ".join(CITIES.keys())
        return f"'{city}'은(는) 지원하지 않는 도시입니다. 사용 가능한 도시: {available_cities}"

    lat, lon = CITIES[city]

    try:
        # Open-Meteo API 사용 (무료, API 키 불필요)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "Asia/Seoul"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        current = data["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind_speed = current["wind_speed_10m"]
        weather_code = current["weather_code"]

        weather_description = get_weather_description(weather_code)

        result = f"""
{city} 날씨 정보
━━━━━━━━━━━━━━━━━━━━
온도: {temp}°C
습도: {humidity}%
풍속: {wind_speed} km/h
날씨: {weather_description}
━━━━━━━━━━━━━━━━━━━━
조회 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        return result.strip()

    except Exception as e:
        return f"날씨 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"


@mcp.tool()
async def list_cities() -> str:
    """
    사용 가능한 도시 목록을 반환합니다.

    Returns:
        사용 가능한 도시 목록
    """
    cities_list = "\n".join([f"  - {city}" for city in CITIES.keys()])
    return f"""
날씨 조회 가능한 도시 목록:
━━━━━━━━━━━━━━━━━━━━
{cities_list}
━━━━━━━━━━━━━━━━━━━━
사용 방법: get_weather(city="도시이름")
    """.strip()


def main():
    """서버 실행 진입점"""
    mcp.run()


if __name__ == "__main__":
    main()