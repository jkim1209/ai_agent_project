# src/agents/financial_analyst.py
"""Financial Analyst Agent - Structured Output 기반 주식 데이터 수집 및 분석

사용자의 질문에 따라 주식 데이터를 수집하고 분석합니다.
ReAct 에이전트 대신 직접 도구를 호출하는 방식으로 구현되었습니다.
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.agents.tools.financial_tools import (
    get_analyst_recommendations,
    get_historical_prices,
    get_korean_ticker,
    get_stock_info,
    search_stocks,
    web_search,
)
from src.model.llm import get_llm_manager
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StockData(BaseModel):
    """개별 주식 데이터 모델 (comparison용)"""
    ticker: str
    company_name: str
    current_price: float
    analysis: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    analyst_recommendation: Optional[str] = None


class AnalysisResult(BaseModel):
    """Financial Analyst의 분석 결과를 위한 Structured Output 모델"""
    analysis_type: Literal["single", "comparison", "concept", "definition", "error"]

    # Single stock analysis fields
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    current_price: Optional[float] = None

    # 공통 필드
    analysis: str = Field(description="분석 내용 또는 설명 (필수)")
    metrics: Optional[Dict[str, Any]] = None
    period: Optional[str] = None
    analyst_recommendation: Optional[str] = None
    historical: Optional[str] = None  # 과거 가격 데이터 (차트 생성용)

    # Comparison fields
    stocks: Optional[List[Dict[str, Any]]] = None
    comparison_summary: Optional[str] = None

    # Concept/Definition fields
    query: Optional[str] = None

    # Error fields
    error: Optional[str] = None


class FinancialAnalyst:
    def __init__(self, model_name: str = None, temperature: float = 0):
        """
        Financial Analyst를 초기화합니다.

        Args:
            model_name: 사용할 모델명 (default: Config.LLM_MODEL)
            temperature: LLM 온도 (0 = 결정적, 1 = 창의적)
        """
        if model_name is None:
            model_name = Config.LLM_MODEL
        logger.info(f"Financial Analyst 초기화 (Structured Output) - model: {model_name}, temp: {temperature}")

        # LLM Manager에서 모델 가져오기
        self.llm_manager = get_llm_manager()
        self.llm = self.llm_manager.get_model(model_name, temperature=temperature)

        logger.info("Financial Analyst 초기화 완료")

    def analyze(self, query: str, messages: list = None, previous_analysis_data: dict = None) -> Dict[str, Any]:
        """
        주어진 질문에 대해 금융 분석을 수행합니다.

        Args:
            query: 사용자 질문
            messages: 대화 히스토리 (선택사항)
            previous_analysis_data: 이전 분석 데이터 (후속 질문 처리용, 선택사항)

        Returns:
            분석 결과를 담은 딕셔너리
        """
        if messages is None:
            messages = []
        if previous_analysis_data is None:
            previous_analysis_data = {}

        try:
            logger.info(f"분석 시작 - query: {query}")

            # Step 1: 질문 분석 및 티커 추출 (이전 분석 티커 제외)
            previous_tickers = self._get_previous_tickers(previous_analysis_data)

            # 제외 전 티커 추출 (동일 종목 재질문 감지용)
            tickers_before_exclusion = self._extract_tickers(query, exclude_tickers=[])
            tickers = self._extract_tickers(query, exclude_tickers=previous_tickers)

            if not tickers:
                # 티커가 없는 경우 2가지 시나리오 구분:
                # 1) 원래 추출 안 됨 → concept query
                # 2) 추출되었으나 전부 제외됨 (동일 종목 재질문) → 재분석
                if tickers_before_exclusion and previous_tickers:
                    # set 비교로 정확한 동일 집합 확인 (순서 무관)
                    if set(tickers_before_exclusion) == set(previous_tickers):
                        # 완전히 동일한 종목 집합 재질문
                        logger.info(f"🔄 동일 종목 재질문 감지: {tickers_before_exclusion} (이전: {previous_tickers})")

                        if len(previous_tickers) >= 2:
                            # 복수 종목 비교 분석 재실행
                            logger.info(f"📊 복수 종목 재분석 모드 (비교 분석): {previous_tickers}")
                            return self._compare_multiple_stocks(previous_tickers, query, messages)
                        else:
                            # 단일 종목 재분석
                            ticker = previous_tickers[0]
                            logger.info(f"📊 단일 종목 재분석 모드: {ticker}")

                            stock_data = self._collect_stock_data(ticker, query)
                            if not stock_data:
                                return {
                                    "analysis_type": "error",
                                    "ticker": ticker,
                                    "company_name": "Unknown",
                                    "current_price": 0,
                                    "analysis": f"{ticker} 주식 정보를 가져올 수 없습니다.",
                                    "error": "데이터 수집 실패"
                                }

                            result = self._generate_analysis(query, stock_data, messages)
                            logger.info(f"✅ 단일 종목 재분석 완료")
                            return result
                    else:
                        # 부분 집합이거나 완전히 다른 경우 → concept query로 처리
                        logger.warning(f"⚠️ 티커 추출됨 ({tickers_before_exclusion})이지만 이전 티커({previous_tickers})와 다름 - 개념 질문으로 처리")
                        return self._handle_concept_query(query)
                else:
                    # 진짜 티커 추출 실패 → concept query
                    logger.warning("티커를 찾을 수 없음 - 개념/정의 질문으로 처리")
                    return self._handle_concept_query(query)

            logger.info(f"✅ 티커 추출 완료: {tickers} (총 {len(tickers)}개)")

            # Step 2: 비교 의도 확인 (LLM 기반)
            has_comparison_intent = self._check_comparison_intent(query)

            # Step 3: 단일 vs 비교 분석 분기
            if len(tickers) >= 2:
                # 티커 2개 이상 = 무조건 비교 분석
                logger.info(f"🔄 비교 분석 모드 (티커 2개 이상): {tickers}")
                return self._compare_multiple_stocks(tickers, query, messages)

            elif len(tickers) == 1 and not has_comparison_intent:
                # 티커 1개 + 비교 의도 없음 = 단일 분석
                ticker = tickers[0]
                logger.info(f"📊 단일 분석 모드: {ticker}")

                stock_data = self._collect_stock_data(ticker, query)

                if not stock_data:
                    return {
                        "analysis_type": "error",
                        "ticker": ticker,
                        "company_name": "Unknown",
                        "current_price": 0,
                        "analysis": f"{ticker} 주식 정보를 가져올 수 없습니다.",
                        "error": "데이터 수집 실패"
                    }

                result = self._generate_analysis(query, stock_data, messages)
                logger.info(f"✅ 단일 분석 완료")
                return result

            elif len(tickers) == 1 and has_comparison_intent:
                # 티커 1개 + 비교 의도 있음 = 이전 분석 데이터 확인
                ticker = tickers[0]
                previous_tickers = self._get_previous_tickers(previous_analysis_data)

                if previous_tickers:
                    # 이전 분석 있음 = 이전 종목 + 현재 종목 비교
                    comparison_tickers = previous_tickers + [ticker]
                    # 중복 제거
                    comparison_tickers = list(dict.fromkeys(comparison_tickers))  # 순서 유지하며 중복 제거

                    logger.info(f"🔄 비교 분석 모드 (이전 분석 활용): {previous_tickers} + {[ticker]} = {comparison_tickers}")
                    return self._compare_multiple_stocks(comparison_tickers, query, messages)
                else:
                    # 이전 분석 없음 = 안내 메시지 + 단일 분석
                    logger.warning(f"⚠️ 비교 의도 감지되었으나 비교 대상이 명시되지 않음")
                    logger.info(f"💡 힌트: 비교하려면 두 개 이상의 종목을 명시해주세요. (예: '삼성전자와 LG전자 비교')")

                    stock_data = self._collect_stock_data(ticker, query)

                    if not stock_data:
                        return {
                            "analysis_type": "error",
                            "ticker": ticker,
                            "company_name": "Unknown",
                            "current_price": 0,
                            "analysis": f"{ticker} 주식 정보를 가져올 수 없습니다.",
                            "error": "데이터 수집 실패"
                        }

                    result = self._generate_analysis(query, stock_data, messages)
                    # 안내 메시지 추가
                    result["analysis"] = "💡 비교 대상이 명시되지 않아 단일 분석을 수행했습니다.\n(예: '삼성전자와 LG전자 비교' 또는 이전 분석 후 'LG와 비교')\n\n" + result.get("analysis", "")
                    logger.info(f"✅ 단일 분석 완료 (비교 의도 있었으나 대상 없음)")
                    return result

            else:
                # 이론상 도달 불가 (티커 0개는 위에서 처리됨)
                logger.error("예상치 못한 분기")
                return self._handle_concept_query(query)

        except Exception as e:
            logger.error(f"분석 실패 - query: {query}, error: {str(e)}")
            import traceback
            logger.debug(f"상세 에러:\n{traceback.format_exc()}")

            return {
                "error": str(e),
                "analysis_type": "error",
                "ticker": "ERROR",
                "company_name": "Error",
                "current_price": 0,
                "analysis": f"분석 중 오류가 발생했습니다: {str(e)}",
                "metrics": {},
                "period": "3mo"
            }

    def _extract_company_names(self, query: str) -> List[str]:
        """
        질문에서 회사명 또는 티커 심볼을 추출합니다 (여러 개 가능).

        Args:
            query: 사용자 질문

        Returns:
            회사명/티커 리스트 (없으면 빈 리스트)
        """
        try:
            # llm.py의 "extract_company_names" 프롬프트 사용
            prompt = self.llm_manager.get_prompt("extract_company_names")
            formatted_prompt = prompt.format_messages(query=query)

            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()

            if content == "NONE" or not content:
                return []

            # 줄바꿈으로 구분된 회사명 파싱
            companies = [line.strip() for line in content.split('\n') if line.strip()]

            # 1단계: 앞쪽 번호/특수문자 제거 (1. 삼성전자 → 삼성전자)
            companies = [c.lstrip('0123456789.-)*# ').strip() for c in companies]

            # 2단계: 필터링
            filtered_companies = []
            for c in companies:
                # 기본 조건: 비어있거나 NONE이면 제외
                if not c or c == "NONE":
                    continue

                # 길이 체크: 2글자 미만 또는 20글자 초과 제외
                if len(c) < 2 or len(c) > 20:
                    logger.debug(f"필터: 길이 제외 - '{c}'")
                    continue

                # 헤더 라인 제외 ("회사명:", "종목:", "Company:" 등)
                if c.endswith(':'):
                    logger.debug(f"필터: 헤더 제외 - '{c}'")
                    continue

                # 설명 문장 키워드 제외
                exclude_keywords = ['회사명', '종목', 'company', '규칙', '답변', '형식', '표기', '예시', '금지', '참고', '티커']
                if any(keyword in c.lower() for keyword in exclude_keywords):
                    logger.debug(f"필터: 설명 문장 제외 - '{c}'")
                    continue

                # 특수문자 많으면 제외 (*, [, ], (, ), 등이 2개 이상)
                special_chars = sum(1 for ch in c if ch in '*[](){}「」『』【】')
                if special_chars >= 2:
                    logger.debug(f"필터: 특수문자 많음 - '{c}'")
                    continue

                # 2글자 티커 중 애매한 것 제외 (KS, NQ 등)
                # 단, 대문자 2글자이고 한글 포함 안 된 경우만
                if len(c) == 2 and c.isupper() and not any('\u3131' <= ch <= '\u318E' or '\uAC00' <= ch <= '\uD7A3' for ch in c):
                    ambiguous_tickers = ['KS', 'NQ', 'US', 'KR', 'JP', 'CN']
                    if c in ambiguous_tickers:
                        logger.debug(f"필터: 애매한 2글자 티커 제외 - '{c}'")
                        continue

                filtered_companies.append(c)

            logger.info(f"✅ 종목/티커 추출: '{query}' → {filtered_companies}")
            return filtered_companies

        except Exception as e:
            logger.error(f"종목/티커 추출 실패: {e}")
            return []

    def _check_comparison_intent(self, query: str) -> bool:
        """LLM을 사용하여 질문이 비교 의도인지 판단

        Args:
            query: 사용자 질문

        Returns:
            비교 의도 여부 (True/False)
        """
        try:
            prompt = self.llm_manager.get_prompt("check_comparison_intent")
            formatted_prompt = prompt.format_messages(query=query)

            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()

            # JSON 파싱
            result = json.loads(content)
            is_comparison = result.get("is_comparison", False)
            reason = result.get("reason", "")

            logger.info(f"비교 의도 판단: {is_comparison} - {reason}")
            return is_comparison

        except Exception as e:
            logger.error(f"비교 의도 판단 실패: {e}")
            # 에러 시 안전하게 False 반환 (단일 분석으로 처리)
            return False

    def _get_previous_tickers(self, previous_analysis_data: dict) -> List[str]:
        """이전 분석 데이터에서 티커 추출

        Args:
            previous_analysis_data: 이전 분석 데이터

        Returns:
            이전 분석한 티커 리스트
        """
        if not previous_analysis_data:
            return []

        analysis_type = previous_analysis_data.get("analysis_type")

        if analysis_type == "single":
            # 단일 분석: ticker 필드 확인
            ticker = previous_analysis_data.get("ticker")
            if ticker:
                logger.info(f"이전 단일 분석 티커 발견: {ticker}")
                return [ticker]

        elif analysis_type == "comparison":
            # 비교 분석: stocks 리스트에서 ticker 추출
            stocks = previous_analysis_data.get("stocks", [])
            tickers = [stock.get("ticker") for stock in stocks if stock.get("ticker")]
            if tickers:
                logger.info(f"이전 비교 분석 티커 발견: {tickers}")
                return tickers

        return []

    def _extract_tickers(self, query: str, exclude_tickers: List[str] = None) -> List[str]:
        """
        질문에서 티커를 추출합니다 (여러 개 가능).

        Args:
            query: 사용자 질문
            exclude_tickers: 추출 결과에서 제외할 티커 리스트 (이전 분석 티커 등)

        Returns:
            티커 리스트 (없으면 빈 리스트)
        """
        if exclude_tickers is None:
            exclude_tickers = []

        try:
            # Step 1: 질문에서 회사명/티커 추출
            company_names = self._extract_company_names(query)
            if not company_names:
                logger.warning("종목명/티커를 추출할 수 없음")
                return []

            # Step 2: 원본 query에서 한글 회사명 추출 (LLM이 "엘지"→"LG"로 번역하는 경우 대비)
            korean_companies = [w for w in re.findall(r'[가-힣]+', query) if len(w) >= 2]
            logger.info(f"원본 쿼리에서 한글 추출: {korean_companies}")

            # 한글 회사명이 있으면 매핑 테이블에서 먼저 검색
            korean_tickers_found = []
            for korean_name in korean_companies:
                korean_ticker = get_korean_ticker(korean_name)
                if korean_ticker:
                    logger.info(f"✅ 한글 매핑 테이블 적용: '{korean_name}' → {korean_ticker}")
                    korean_tickers_found.append((korean_name, korean_ticker))

            # Step 3: 각 종목명/티커로 티커 검색
            tickers = []
            used_korean_names = set()

            for company_name in company_names:
                logger.info(f"티커 검색 중: {company_name}")

                # 먼저 한글 매핑에서 찾은 것 중에 company_name과 일치하는 것 우선 사용
                matched_ticker = None
                for korean_name, korean_ticker in korean_tickers_found:
                    if korean_name not in used_korean_names:
                        # company_name과 korean_name이 유사한지 확인
                        # 예: "LG전자"와 "LG전자" 매칭, "삼성전자"와 "삼성전자를" 매칭
                        company_normalized = company_name.lower().replace(" ", "")
                        korean_normalized = korean_name.lower().replace(" ", "")

                        # 직접 매칭하거나, 한쪽이 다른 쪽을 포함하는 경우
                        if (company_normalized in korean_normalized or
                            korean_normalized in company_normalized):
                            matched_ticker = korean_ticker
                            used_korean_names.add(korean_name)
                            logger.info(f"✅ 한글 매핑 우선 사용: '{korean_name}' → {matched_ticker} ('{company_name}'와 매칭)")
                            break

                if matched_ticker and matched_ticker not in tickers:
                    tickers.append(matched_ticker)
                    continue

                # 매핑 안 되면 search_stocks로 검색 (거래소 우선순위 적용)
                result = search_stocks.invoke({"query": company_name, "max_results": 5})

                if "찾을 수 없습니다" in result or "오류" in result:
                    logger.warning(f"티커 검색 실패: {company_name}")
                    continue

                # 결과에서 첫 번째 티커 추출 (거래소 우선순위 적용된 결과)
                # 포맷: "• TICKER - Company Name [EXCHANGE]"
                matches = re.findall(r'•\s*([A-Z0-9.]+)\s*-\s*([^\[]+)\[([^\]]+)\]', result)

                if matches:
                    # 첫 번째 티커 (우선순위 가장 높음)
                    ticker, name, exchange = matches[0]
                    ticker = ticker.strip()
                    name = name.strip()
                    exchange = exchange.strip()

                    # 중복 체크
                    if ticker not in tickers:
                        logger.info(f"✅ 티커 추출 성공: {ticker} ({name}) [{exchange}]")

                        # 여러 후보가 있으면 로그에 기록
                        if len(matches) > 1:
                            other_candidates = [f"{t.strip()} ({n.strip()}) [{e.strip()}]"
                                              for t, n, e in matches[1:]]
                            logger.info(f"📋 다른 후보: {', '.join(other_candidates[:3])}")
                            if "여러 후보가 있습니다" in result:
                                logger.info(f"⚠️ '{company_name}' 검색 시 여러 후보 발견 - 거래소 우선순위 적용하여 {ticker} 선택")

                        tickers.append(ticker)
                    else:
                        logger.info(f"⚠️ {ticker}는 이미 추출된 티커 (중복 제거)")
                else:
                    logger.warning(f"티커 파싱 실패 - result: {result[:200]}")

            # exclude_tickers 필터링
            if exclude_tickers:
                original_count = len(tickers)
                tickers = [t for t in tickers if t not in exclude_tickers]
                if original_count != len(tickers):
                    logger.info(f"🔄 이전 분석 티커 제외: {original_count}개 → {len(tickers)}개 (제외: {[t for t in exclude_tickers if t in tickers[:original_count]]})")

            return tickers

        except Exception as e:
            logger.error(f"티커 추출 실패: {e}")
            return []

    def _collect_stock_data(self, ticker: str, query: str) -> Optional[Dict[str, Any]]:
        """
        티커에 대한 모든 데이터를 수집합니다.

        Args:
            ticker: 주식 티커
            query: 사용자 질문

        Returns:
            수집된 데이터 딕셔너리
        """
        try:
            collected_data = {"ticker": ticker}

            # 1. 주식 기본 정보
            logger.info(f"📊 주식 정보 조회: {ticker}")
            try:
                stock_info = get_stock_info.invoke({"ticker": ticker})
                collected_data["stock_info"] = stock_info
                logger.info(f"✅ 주식 정보 수집 완료")
            except Exception as e:
                logger.warning(f"⚠️ 주식 정보 수집 실패: {e}")
                collected_data["stock_info"] = {}

            # 2. 과거 가격 데이터
            logger.info(f"📈 과거 가격 데이터 조회: {ticker}")
            try:
                historical = get_historical_prices.invoke({"ticker": ticker, "period": "3mo", "interval": "1d"})
                collected_data["historical"] = historical
                logger.info(f"✅ 과거 데이터 수집 완료")

                # 52주 최고가/최저가가 없으면 과거 데이터에서 계산
                stock_info = collected_data.get("stock_info", {})
                if (stock_info.get("52week_high", 0) == 0 or stock_info.get("52week_low", 0) == 0) and historical:
                    try:
                        # historical 데이터 파싱 (CSV 형식 또는 딕셔너리)
                        import pandas as pd
                        if isinstance(historical, str):
                            from io import StringIO
                            # 첫 줄은 메타데이터, 그 다음부터 CSV
                            lines = historical.strip().split('\n')
                            if len(lines) > 1:
                                csv_data = '\n'.join(lines[1:])
                                df = pd.read_csv(StringIO(csv_data))
                            else:
                                df = pd.DataFrame()
                        elif isinstance(historical, dict):
                            df = pd.DataFrame(historical)
                        else:
                            df = historical

                        if not df.empty and 'High' in df.columns and 'Low' in df.columns:
                            high_52w = df['High'].max()
                            low_52w = df['Low'].min()

                            # stock_info 업데이트
                            if stock_info.get("52week_high", 0) == 0:
                                stock_info["52week_high"] = high_52w
                                logger.info(f"✅ 52주 최고가 계산: {high_52w:.2f}")

                            if stock_info.get("52week_low", 0) == 0:
                                stock_info["52week_low"] = low_52w
                                logger.info(f"✅ 52주 최저가 계산: {low_52w:.2f}")

                            collected_data["stock_info"] = stock_info
                    except Exception as calc_err:
                        logger.warning(f"⚠️ 52주 데이터 계산 실패: {calc_err}")
            except Exception as e:
                logger.warning(f"⚠️ 과거 데이터 수집 실패: {e}")
                collected_data["historical"] = ""

            # 3. 웹 검색 (뉴스/분석)
            logger.info(f"🔍 웹 검색: {query}")
            try:
                web_result = web_search.invoke({"query": f"{ticker} stock news analysis"})
                collected_data["web_search"] = web_result
                logger.info(f"✅ 웹 검색 완료")
            except Exception as e:
                logger.warning(f"⚠️ 웹 검색 실패: {e}")
                collected_data["web_search"] = ""

            # 4. 애널리스트 추천
            logger.info(f"💼 애널리스트 추천 조회: {ticker}")
            try:
                analyst_rec = get_analyst_recommendations.invoke({"ticker": ticker})
                collected_data["analyst_rec"] = analyst_rec
                logger.info(f"✅ 애널리스트 추천 수집 완료")
            except Exception as e:
                logger.warning(f"⚠️ 애널리스트 추천 수집 실패: {e}")
                collected_data["analyst_rec"] = ""

            return collected_data

        except Exception as e:
            logger.error(f"데이터 수집 실패: {e}")
            return None

    def _compare_multiple_stocks(
        self,
        tickers: List[str],
        query: str,
        messages: list
    ) -> Dict[str, Any]:
        """
        여러 주식을 비교 분석합니다.

        Args:
            tickers: 티커 리스트
            query: 사용자 질문
            messages: 대화 히스토리

        Returns:
            비교 분석 결과 딕셔너리
        """
        try:
            logger.info(f"📊 {len(tickers)}개 종목 데이터 수집 시작")

            # Step 1: 각 티커별로 데이터 수집
            stocks_data = []
            for ticker in tickers:
                logger.info(f"📈 {ticker} 데이터 수집 중...")
                stock_data = self._collect_stock_data(ticker, query)

                if stock_data:
                    stock_info = stock_data.get("stock_info", {})

                    # metrics를 stock_info 전체 데이터로 구성 (중복 제거)
                    metrics = {
                        "pe_ratio": stock_info.get("pe_ratio"),
                        "forward_pe": stock_info.get("forward_pe"),
                        "pb_ratio": stock_info.get("pb_ratio"),
                        "market_cap": stock_info.get("market_cap", 0),
                        "dividend_yield": stock_info.get("dividend_yield", 0),
                        "52week_high": stock_info.get("52week_high", 0),
                        "52week_low": stock_info.get("52week_low", 0),
                        "volume": stock_info.get("volume", 0),
                        "avg_volume": stock_info.get("avg_volume", 0),
                        "sector": stock_info.get("sector", "N/A"),
                        "industry": stock_info.get("industry", "N/A")
                    }

                    stocks_data.append({
                        "ticker": ticker,
                        "company_name": stock_info.get("name", "Unknown"),
                        "current_price": stock_info.get("current_price", 0),
                        "metrics": metrics,
                        "historical": stock_data.get("historical", ""),  # 차트 생성용
                        "data": stock_data  # 전체 데이터 보관
                    })
                    logger.info(f"✅ {ticker} 데이터 수집 완료")
                else:
                    logger.warning(f"⚠️ {ticker} 데이터 수집 실패")

            if not stocks_data:
                return {
                    "analysis_type": "error",
                    "stocks": [],
                    "analysis": "모든 종목의 데이터 수집에 실패했습니다.",
                    "error": "데이터 수집 실패"
                }

            # Step 2: Structured Output으로 비교 분석 생성
            logger.info("🤖 비교 분석 생성 중...")
            result = self._generate_comparison_analysis(query, stocks_data, messages)

            logger.info(f"✅ 비교 분석 완료 - {len(stocks_data)}개 종목")
            return result

        except Exception as e:
            logger.error(f"비교 분석 실패: {e}")
            return {
                "analysis_type": "error",
                "stocks": [],
                "analysis": f"비교 분석 중 오류 발생: {str(e)}",
                "error": str(e)
            }

    def _generate_comparison_analysis(
        self,
        query: str,
        stocks_data: List[Dict[str, Any]],
        messages: list
    ) -> Dict[str, Any]:
        """
        여러 종목의 비교 분석을 생성합니다.

        Args:
            query: 사용자 질문
            stocks_data: 각 종목의 수집된 데이터 리스트
            messages: 대화 히스토리

        Returns:
            비교 분석 결과 딕셔너리
        """
        # 각 종목 요약 (폴백용으로도 사용)
        stocks_summary = []
        for stock in stocks_data:
            stocks_summary.append({
                "ticker": stock["ticker"],
                "company_name": stock["company_name"],
                "current_price": stock["current_price"],
                "metrics": stock.get("metrics", {})
            })

        try:
            # Structured Output 설정
            llm_with_structure = self.llm.with_structured_output(AnalysisResult)

            # llm.py의 "analyze_comparison" 프롬프트 사용
            prompt = self.llm_manager.get_prompt("analyze_comparison")
            formatted_prompt = prompt.format_messages(
                query=query,
                stocks_summary=json.dumps(stocks_summary, ensure_ascii=False, indent=2)
            )

            # Structured Output으로 분석 생성
            result = llm_with_structure.invoke(formatted_prompt)

            # Pydantic 모델을 딕셔너리로 변환
            result_dict = result.model_dump()

            # stocks를 historical 포함된 stocks_data로 교체
            result_dict["stocks"] = stocks_data

            return result_dict

        except Exception as e:
            logger.error(f"비교 분석 생성 실패: {e}")

            # 폴백: 기본 구조로 반환
            return {
                "analysis_type": "comparison",
                "stocks": stocks_data,  # historical 포함된 stocks_data 사용
                "analysis": f"{len(stocks_data)}개 종목의 데이터를 수집했으나 비교 분석 생성에 실패했습니다.",
                "comparison_summary": "분석 생성 실패"
            }

    def _generate_analysis(
        self,
        query: str,
        stock_data: Dict[str, Any],
        messages: list
    ) -> Dict[str, Any]:
        """
        수집된 데이터를 기반으로 최종 분석을 생성합니다 (Structured Output).

        Args:
            query: 사용자 질문
            stock_data: 수집된 주식 데이터
            messages: 대화 히스토리

        Returns:
            AnalysisResult 딕셔너리
        """
        try:
            # Structured Output 설정
            llm_with_structure = self.llm.with_structured_output(AnalysisResult)

            # 데이터 요약
            ticker = stock_data.get("ticker", "UNKNOWN")
            stock_info = stock_data.get("stock_info", {})
            company_name = stock_info.get("name", stock_info.get("company_name", "Unknown"))
            current_price = stock_info.get("current_price", 0)

            # metrics를 stock_info에서 직접 구성 (중복 제거)
            metrics = {
                "pe_ratio": stock_info.get("pe_ratio"),
                "forward_pe": stock_info.get("forward_pe"),
                "pb_ratio": stock_info.get("pb_ratio"),
                "market_cap": stock_info.get("market_cap", 0),
                "dividend_yield": stock_info.get("dividend_yield", 0),
                "52week_high": stock_info.get("52week_high", 0),
                "52week_low": stock_info.get("52week_low", 0),
                "volume": stock_info.get("volume", 0),
                "avg_volume": stock_info.get("avg_volume", 0),
                "sector": stock_info.get("sector", "N/A"),
                "industry": stock_info.get("industry", "N/A")
            }

            # historical 데이터 정보 추출
            historical_info = "없음"
            historical_data = stock_data.get('historical', '')
            if historical_data and len(historical_data.strip()) > 0:
                # 첫 줄에서 메타데이터 추출 (예: "005930.KS 과거 가격 (3mo, 1d 간격) - 총 60개 데이터 포인트")
                first_line = historical_data.strip().split('\n')[0]
                historical_info = f"수집 완료 ({first_line})"

            # llm.py의 "analyze_single_stock" 프롬프트 사용
            prompt = self.llm_manager.get_prompt("analyze_single_stock")
            formatted_prompt = prompt.format_messages(
                company_name=company_name,
                ticker=ticker,
                query=query,
                current_price=current_price,
                metrics=json.dumps(metrics, ensure_ascii=False)[:500],
                historical_info=historical_info,
                web_search=str(stock_data.get('web_search', ''))[:500],
                analyst_rec=str(stock_data.get('analyst_rec', ''))[:300]
            )

            # Structured Output으로 분석 생성
            result = llm_with_structure.invoke(formatted_prompt)

            # Pydantic 모델을 딕셔너리로 변환
            result_dict = result.model_dump()

            # historical 데이터 추가 (차트 생성용)
            result_dict["historical"] = stock_data.get("historical", "")

            # metrics를 실제 수집된 데이터로 덮어쓰기 (LLM이 잘못 생성한 경우 방지)
            result_dict["metrics"] = metrics

            return result_dict

        except Exception as e:
            logger.error(f"분석 생성 실패: {e}")

            # 폴백: 기본 구조로 반환
            stock_info = stock_data.get("stock_info", {})

            # metrics 구성 (stock_info는 평탄한 구조)
            fallback_metrics = {
                "pe_ratio": stock_info.get("pe_ratio"),
                "forward_pe": stock_info.get("forward_pe"),
                "pb_ratio": stock_info.get("pb_ratio"),
                "market_cap": stock_info.get("market_cap", 0),
                "dividend_yield": stock_info.get("dividend_yield", 0),
                "52week_high": stock_info.get("52week_high", 0),
                "52week_low": stock_info.get("52week_low", 0),
                "volume": stock_info.get("volume", 0),
                "avg_volume": stock_info.get("avg_volume", 0),
                "sector": stock_info.get("sector", "N/A"),
                "industry": stock_info.get("industry", "N/A")
            }

            return {
                "analysis_type": "single",
                "ticker": stock_data.get("ticker", "UNKNOWN"),
                "company_name": stock_info.get("company_name", "Unknown"),
                "current_price": stock_info.get("current_price", 0),
                "analysis": f"{stock_info.get('company_name', 'Unknown')} 주식에 대한 분석 데이터를 수집했습니다.",
                "metrics": fallback_metrics,
                "historical": stock_data.get("historical", ""),
                "period": "3mo",
                "analyst_recommendation": "N/A"
            }

    def _handle_concept_query(self, query: str) -> Dict[str, Any]:
        """
        개념/정의 질문을 처리합니다 (티커 없는 경우).

        Args:
            query: 사용자 질문

        Returns:
            AnalysisResult 딕셔너리
        """
        try:
            logger.info(f"개념 질문 처리: {query}")

            # llm.py의 "analyze_concept" 프롬프트 사용
            prompt = self.llm_manager.get_prompt("analyze_concept")
            formatted_prompt = prompt.format_messages(query=query)

            response = self.llm.invoke(formatted_prompt)
            explanation = response.content.strip()

            return {
                "analysis_type": "concept",
                "query": query,
                "analysis": explanation
            }

        except Exception as e:
            logger.error(f"개념 질문 처리 실패: {e}")
            return {
                "analysis_type": "error",
                "query": query,
                "analysis": f"질문을 처리할 수 없습니다: {str(e)}",
                "error": str(e)
            }

    def compare_stocks(self, tickers: List[str], messages: list = None) -> Dict[str, Any]:
        """
        여러 주식을 비교 분석합니다.

        Args:
            tickers: 비교할 티커 리스트 (예: ["AAPL", "MSFT", "GOOGL"])
            messages: 대화 히스토리

        Returns:
            비교 분석 결과 딕셔너리
        """
        if messages is None:
            messages = []

        try:
            logger.info(f"비교 분석 시작 - tickers: {tickers}")

            # 자동으로 비교 쿼리 생성
            ticker_str = ", ".join(tickers)
            query = f"{ticker_str} 주식들을 비교 분석해주세요. 각각의 장단점과 투자 추천을 포함해주세요."

            return self.analyze(query=query, messages=messages)

        except Exception as e:
            logger.error(f"비교 분석 실패 - tickers: {tickers}, error: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": "comparison",
                "stocks": [],
                "comparison_analysis": f"비교 분석 중 오류가 발생했습니다: {str(e)}"
            }

    def invoke(self, query: str, messages: list = None) -> Dict[str, Any]:
        """
        analyze()의 별칭 메서드 (LangChain 스타일 호환)

        Args:
            query: 사용자 질문
            messages: 대화 히스토리

        Returns:
            분석 결과 딕셔너리
        """
        return self.analyze(query=query, messages=messages)


# 편의를 위한 팩토리 함수
def create_financial_analyst(
    model_name: str = "solar-pro",
    temperature: float = 0
) -> FinancialAnalyst:
    """
    Financial Analyst를 생성합니다.

    Args:
        model_name: 사용할 LLM 모델명
        temperature: LLM 온도

    Returns:
        FinancialAnalyst 인스턴스
    """
    return FinancialAnalyst(model_name=model_name, temperature=temperature)


if __name__ == "__main__":
    import logging

    # 디버그 로그 활성화
    logging.getLogger("__main__").setLevel(logging.DEBUG)
    logging.getLogger("langchain.agents.agent").setLevel(logging.ERROR)

    from src.utils.config import Config
    Config.validate_api_keys()

    analyst = create_financial_analyst(model_name="solar-pro")

    # 단일 분석
    print("\n" + "="*80)
    print("단일 주식 분석")
    print("="*80)
    result = analyst.analyze("애플 주식 분석")
    print(f"분석 타입: {result.get('analysis_type')}")
    print(f"티커: {result.get('ticker')}")
    print(f"분석: {result.get('analysis', '')[:200]}...")
