# src/workflow/workflow.py
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict, Annotated

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agents.financial_analyst import FinancialAnalyst
from src.evaluator.llm_quality_evaluator import QualityEvaluator
from src.agents.report_generator import ReportGenerator
from src.agents.request_analyst import request_analysis, rewrite_query
from src.agents.supervisor import supervisor
from src.agents.query_cleaner import query_cleaner
from src.model.llm import get_llm_manager
from src.rag.retriever import Retriever
from src.utils.config import Config

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowState(TypedDict, total=False):
    """LangGraph 워크플로우에서 사용하는 상태 구조.

    노드 간에 전달되는 모든 데이터를 포함하며, 질문 분석부터 답변 생성, 품질 평가까지
    전체 워크플로우의 상태를 추적합니다.
    """
    session_id: str # 사용자 세션 id
    question: str # 사용자의 질문
    answer: str   # LLM 의 생성 답변
    route: Literal["end", "supervisor", "financial_analyst", "report_generator"]
    request_type: Literal["rag", "financial_analyst"]  # report_generator 의 2가지 task 분기
    rag_search_results: List[str]  # Rag 의 검색 결과
    analysis_data: Dict[str, object] # Rag 혹은 financial_analyst 의 최종 분석 결과
    quality_passed: bool       # quality_evaluator 에서의 품질 통과 여부
    quality_detail: Dict[str, object]  # quality_evaluator 의 평가 결과 디테일
    retries : int  # 루프 재시도 횟수
    previous_failure_reason: str  # 이전 실패 이유 (연속 실패 감지용)
    consecutive_same_failures: int  # 동일 실패 연속 횟수
    messages : Annotated[list, add_messages]
    agent_scratchpad : Annotated[list, add_messages]
    # 현재 응답에서 생성된 파일 (streamlit 표시용)
    current_charts: List[str]
    current_saved_file: str
    


class Workflow:
    """요청 분석 → 라우팅 → 답변 생성 → 품질 평가 → (선택:재시도 루프) 까지 이어지는 워크플로우."""

    def __init__(self):
        self.llm_manager = get_llm_manager()
        self.shared_llm = self.llm_manager.get_model(Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)

        self.retriever = Retriever()
        self.financial_analyst = FinancialAnalyst()
        self.report_generator = ReportGenerator()
        self.quality_evaluator = QualityEvaluator(llm=self.shared_llm)
        self.graph = self._build_graph()

   
    def _build_graph(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("query_clean", self.query_clean_node)
        graph.add_node("request_analyst", self.request_analyst_node)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("financial_analyst", self.financial_analyst_node)
        graph.add_node("general_conversation", self.general_conversation_node)
        graph.add_node("report_generator", self.report_generator_node)
        graph.add_node("quality_evaluator", self.quality_evaluator_node)

        graph.set_entry_point("query_clean")

        # query_clean → request_analyst (모든 질문에 대해 오타 수정 먼저)
        graph.add_edge("query_clean", "request_analyst")

        graph.add_conditional_edges(
            "request_analyst",
            self._route_from_request_analyst,
            {
                "end": END,
                "supervisor": "supervisor",
                "report_generator": "report_generator",
                "general_conversation": "general_conversation",
            },
        )

        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "financial_analyst": "financial_analyst",
                "report_generator": "report_generator",
                "general_conversation": "general_conversation",
                "end": END,
            },
        )

        graph.add_edge("financial_analyst", "report_generator")
        graph.add_edge("general_conversation", END)  # 일반 대화는 바로 종료
        graph.add_edge("report_generator", "quality_evaluator")

        graph.add_conditional_edges(
            "quality_evaluator",
            self._route_from_quality_evaluator,
            {
                "retry": "request_analyst",
                "end": END,
            },
        )

        return graph.compile()

    # ------------------------------------------------------------------ #
    # Node 
    # ------------------------------------------------------------------ #
    def request_analyst_node(self, state: WorkflowState) -> WorkflowState:
        """사용자 요청을 분석하여 finance/general_conversation/not_finance 3가지로 분류합니다.

        후속 질문(차트/PDF 저장 요청)을 감지하여 이전 분석 데이터가 있으면 report_generator로 직접 라우팅합니다.
        - finance: supervisor로 라우팅
        - general_conversation: 일반 대화 노드로 라우팅
        - not_finance: 안내 메시지와 함께 종료
        """
        question = state.get("question", "").strip()
        if not question:
            state["answer"] = "질문이 비어 있어 답변을 드릴 수 없습니다."
            state["route"] = "end"
            return state

        # 후속 질문 감지 (PDF 저장, 차트 생성 등)
        has_previous_analysis = state.get("analysis_data") is not None

        # 후속 질문 키워드 (차트/저장만 요청)
        follow_up_keywords = ["차트", "그래프", "저장", "그려", "다운로드", "파일", "pdf", "md", "markdown"]
        has_follow_up_keyword = any(keyword in question.lower() for keyword in follow_up_keywords)

        # 새로운 분석 요청 키워드 (새로운 작업)
        new_request_keywords = ["분석", "비교", "알려줘", "조회", "찾아", "검색", "주식", "기업", "회사"]
        # 예외: "분석 결과", "분석내용", "보고서" 등은 기존 결과를 참조하는 것이므로 새로운 요청이 아님
        exception_patterns = ["분석 결과", "분석결과", "분석내용", "분석 내용", "보고서", "리포트", "비교 분석", "비교분석"]
        has_exception = any(pattern in question for pattern in exception_patterns)
        has_new_request = any(keyword in question.lower() for keyword in new_request_keywords) and not has_exception

        # 추가 요청 패턴 (~도, ~까지, ~포함)
        additional_patterns = ["도 ", "까지", "포함", "추가", "더"]
        has_additional_pattern = any(pattern in question for pattern in additional_patterns)

        # 후속 질문 판단: 키워드 있고 + 새로운 요청 없고 + 추가 요청 패턴 없음
        is_follow_up = has_follow_up_keyword and not has_new_request and not has_additional_pattern

        if has_previous_analysis and is_follow_up:
            logger.info(f"📊 후속 질문 감지 (request_analyst 우회) - 이전 분석 데이터로 바로 report_generator 호출")
            state["route"] = "report_generator"
            state["request_type"] = "financial_analyst"
            return state

        # 일반적인 금융 질문 분석
        analysis_result = request_analysis(state, llm=self.shared_llm)
        label = analysis_result.get("label")

        if label == "finance":
            state["route"] = "supervisor"
        elif label == "general_conversation":
            # 일반 대화는 general_conversation_node로 라우팅
            state["route"] = "general_conversation"
        else:  # not_finance
            # 비금융 질문인 경우 안내 메시지를 그대로 전달
            state["answer"] = analysis_result.get("return_msg", "경제, 금융관련 질문이 아닙니다.")
            state["quality_passed"] = True  # 정상 응답이므로 success로 저장
            state["route"] = "end"
        return state

    def supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """슈퍼바이저 에이전트를 호출해 다음 노드를 결정합니다."""
        # 일반적인 라우팅
        agent_choice = supervisor(
            state,
            llm=self.shared_llm,
        )

        if agent_choice == "financial_analyst":
            state["route"] = "financial_analyst"
        elif agent_choice == "vector_search_agent":
            state["route"] = "report_generator"
            state["request_type"] = "rag"
        else:  # "none" - 일반 대화, 인사, 메타 질문 등
            logger.info("💬 일반 대화로 라우팅 (general_conversation)")
            state["route"] = "general_conversation"
        return state

    def financial_analyst_node(self, state: WorkflowState) -> WorkflowState:
        """재무 분석 에이전트를 실행 후, report_generator를 호출합니다."""
        question = state.get("question", "")
        messages = state.get('messages', [])
        logger.info(f"🔍 financial_analyst_node 시작")

        try:
            analysis_data = self.financial_analyst.analyze(
                query=question,
                messages=messages,
                previous_analysis_data=state.get("analysis_data")
            )
            # 중요: 반환값 확인
            logger.info(f"📊 analyze() 반환 타입: {type(analysis_data)}")
            logger.debug(f"📊 analyze() 반환 값: {analysis_data}")

            # 데이터 유효성 검증
            if not analysis_data or not isinstance(analysis_data, dict):
                logger.error("❌ financial_analyst가 유효하지 않은 데이터 반환")
                state["answer"] = "죄송합니다. 주식 분석 중 문제가 발생했습니다. 다시 시도해주세요."
                state["route"] = "end"
                return state

            state["analysis_data"] = analysis_data
            state["request_type"] = "financial_analyst"
            logger.info(f"✅ state에 저장 완료")
            logger.debug(f"✅ state['analysis_data'] 확인: {state.get('analysis_data', 'NOT FOUND')}")

        except Exception as e:
            logger.error(f"❌ financial_analyst_node 실행 중 오류: {e}", exc_info=True)
            state["answer"] = f"주식 분석 중 오류가 발생했습니다: {str(e)}"
            state["route"] = "end"

        return state

    def query_clean_node(self, state: WorkflowState) -> WorkflowState:
        """모든 사용자 질문의 오탈자를 수정하고 대화 문맥을 파악하여 정확한 쿼리로 재작성합니다.

        에러 발생 시에도 원본 질문을 유지하며 워크플로우를 계속 진행합니다.
        """
        question = state.get("question", "")
        messages = state.get('messages', [])
        logger.info("="*10 + " Query Clean node 진입 " + "="*10)
        logger.info(f"원본 질문: {question}")

        try:
            result = query_cleaner({'question': question, 'messages': messages}, llm=self.shared_llm)
            rewritten_query = result.get('rewritten_query', question)

            # 원본과 다른 경우에만 로그 출력
            if rewritten_query != question:
                logger.info(f"✨ 쿼리 정제 완료: {question} → {rewritten_query}")
                state['question'] = rewritten_query
            else:
                logger.info("ℹ️ 쿼리 변경 불필요 (원본 그대로 사용)")

        except Exception as e:
            logger.error(f"❌ query_cleaner 실행 중 오류 발생: {e}", exc_info=True)
            logger.warning("⚠️ 쿼리 정제 실패 - 원본 질문 그대로 사용")
            # 에러 발생 시 원본 질문 유지

        # graph.add_edge로 자동 연결되므로 route 설정 불필요
        return state

    def general_conversation_node(self, state: WorkflowState) -> WorkflowState:
        """일반 대화, 인사, 감사, 메타 질문을 3단계로 처리합니다.

        1) 규칙 기반 패턴 매칭으로 빠른 응답 (인사, 감사, 작별)
        2) 대화 히스토리를 참조한 메타 질문 처리 ("방금 뭐 물어봤지?")
        3) 복잡한 경우 LLM 기반 응답 생성

        모든 응답은 quality_passed=True로 설정되어 품질 평가를 우회합니다.
        """
        question = state.get("question", "").strip()
        question_lower = question.lower()
        messages = state.get("messages", [])

        logger.info(f"💬 general_conversation_node 시작 - question: {question}")

        # 1단계: 규칙 기반 패턴 매칭 (빠른 응답, LLM 비용 절감)
        greetings = ["안녕", "하이", "hi", "hello", "헬로"]
        thanks = ["고마", "감사", "thanks", "thank you", "땡큐"]
        goodbyes = ["잘가", "안녕히", "bye", "goodbye", "바이"]

        if any(g in question_lower for g in greetings):
            state["answer"] = "안녕하세요! 금융 관련 궁금한 점이 있으시면 언제든 물어보세요. 📊"
            state["quality_passed"] = True  # 정상 응답
            state["route"] = "end"
            logger.info("💬 규칙 기반 응답: 인사")
            return state

        if any(t in question_lower for t in thanks):
            state["answer"] = "도움이 되었다니 기쁩니다! 다른 궁금한 점이 있으시면 말씀해주세요. 😊"
            state["quality_passed"] = True  # 정상 응답
            state["route"] = "end"
            logger.info("💬 규칙 기반 응답: 감사")
            return state

        if any(gb in question_lower for gb in goodbyes):
            state["answer"] = "좋은 하루 되세요! 언제든 다시 찾아주세요. 👋"
            state["quality_passed"] = True  # 정상 응답
            state["route"] = "end"
            logger.info("💬 규칙 기반 응답: 작별")
            return state

        # 2단계: 메타 질문 처리 (대화 히스토리 참조)
        from langchain_core.messages import HumanMessage, AIMessage
        meta_patterns = ["방금", "아까", "전에", "처음", "첫", "이전"]

        if any(mp in question_lower for mp in meta_patterns):
            # messages에서 HumanMessage만 추출
            user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]

            if len(user_messages) >= 1:  # 이전 메시지가 있으면
                prev_question = user_messages[-1].content  # 가장 최근 사용자 질문
                state["answer"] = f'방금 물어보신 질문은 "{prev_question}" 입니다.'
                state["quality_passed"] = True  # 정상 응답
                state["route"] = "end"
                logger.info(f"💬 메타 질문 처리: 이전 질문 인용 - {prev_question[:50]}")
                return state
            else:
                state["answer"] = "이전 질문이 없습니다. 지금 처음 대화를 시작하신 것 같네요!"
                state["quality_passed"] = True  # 정상 응답
                state["route"] = "end"
                logger.info("💬 메타 질문 처리: 이전 질문 없음")
                return state

        # 3단계: LLM 기반 일반 대화 (복잡한 경우)
        try:
            logger.info("💬 LLM 기반 일반 대화 처리 시작")
            llm_manager = get_llm_manager()
            llm = llm_manager.get_model("solar-mini", temperature=0.7)
            prompt = llm_manager.get_prompt("general_conversation")

            # 프롬프트 체인 실행 (MessagesPlaceholder가 자동으로 처리)
            chain = prompt | llm
            response = chain.invoke({"input": question, "chat_history": messages})

            state["answer"] = response.content.strip()
            state["quality_passed"] = True  # 정상 응답
            state["route"] = "end"
            logger.info(f"💬 LLM 응답 생성 완료 - 길이: {len(state['answer'])}자")

        except Exception as e:
            logger.error(f"❌ general_conversation_node LLM 처리 실패: {e}")
            state["answer"] = "죄송합니다. 응답 생성 중 문제가 발생했습니다. 다시 시도해주세요."
            state["quality_passed"] = True  # 에러 메시지도 정상 응답으로 처리
            state["route"] = "end"

        return state

    def report_generator_node(self, state: WorkflowState) -> WorkflowState:
        """RAG 검색 또는 financial_analyst의 분석 결과를 바탕으로 최종 보고서를 생성합니다.

        RAG 검색 결과가 없으면 financial_analyst로 자동 폴백합니다.
        생성된 차트와 파일 정보를 state의 current_charts, current_saved_file 필드에 저장하여
        streamlit에서 표시할 수 있도록 합니다.
        """
        question = state.get("question", "")
        messages = state.get('messages', [])
        logger.info(f"📝 report_generator_node 진입")
        logger.info(f"📝 request_type: {state.get('request_type', 'NOT SET')}")
        
        if state.get("request_type","rag") == "rag":
            logger.info("📝 RAG 모드")
            results = self.retriever.retrieve(question)

            # RAG 검색 결과가 없을 때 financial_analyst로 폴백
            if not results or len(results) == 0:
                logger.warning("⚠️ RAG 검색 결과가 없습니다. financial_analyst로 폴백 시도...")

                # financial_analyst를 직접 호출해서 웹 검색 시도
                try:
                    analysis_data = self.financial_analyst.analyze(
                        query=question,
                        messages=messages,
                        previous_analysis_data=state.get("analysis_data")
                    )

                    if analysis_data and isinstance(analysis_data, dict):
                        logger.info("✅ financial_analyst 폴백 성공")
                        state["analysis_data"] = analysis_data
                        state["request_type"] = "financial_analyst"
                        # 이제 아래 else 블록에서 처리됨
                    else:
                        logger.error("❌ financial_analyst 폴백 실패 - 분석 데이터 없음")
                        state["answer"] = (
                            "죄송합니다. 데이터베이스와 웹 검색 모두에서 관련 정보를 찾을 수 없습니다.\n\n"
                            "다른 주제로 질문해주시거나, 질문을 더 구체적으로 작성해주세요."
                        )
                        return state

                except Exception as e:
                    logger.error(f"❌ financial_analyst 폴백 중 오류: {e}", exc_info=True)
                    state["answer"] = (
                        "죄송합니다. 정보를 찾는 과정에서 오류가 발생했습니다.\n"
                        "잠시 후 다시 시도해주세요."
                    )
                    return state
            else:
                # RAG 검색 결과가 있는 경우
                rag_search_results = []
                for doc, score in results:
                    page = doc.metadata.get("page", "?")
                    if isinstance(page, int):
                        page += 1  # 0-index → 1-index 변환
                    source = doc.metadata.get("source", "unknown")
                    rag_search_results.append(f"- (score={score:.2f}) {source} p.{page}")

                state["rag_search_results"] = rag_search_results

                analysis_data = {
                "analysis_type" : "rag",
                "query": question,
                "documents": [doc.page_content for doc, _ in results],
                }

                state["analysis_data"] = analysis_data

        else:
            # financial_analyst 에서 호출 시, 해당 분석 결과 사용
            logger.info("📝 financial_analyst 모드")
            analysis_data = state.get("analysis_data")
            if not analysis_data:
                logger.error("❌ analysis_data가 state에 없습니다!")
                state["answer"] = "분석 데이터를 찾을 수 없습니다. 다시 시도해주세요."
                return state

            logger.debug(f"✅ State 저장소 analysis_data 로드: {analysis_data.get('analysis_type', 'N/A')}")

        # 보고서 생성 with 에러 처리
        try:
            report = self.report_generator.generate_report(user_request=question, analysis_data=analysis_data, messages = messages)

            if not report or not isinstance(report, dict):
                logger.error("❌ report_generator가 유효하지 않은 데이터 반환")
                state["answer"] = "보고서 생성 중 문제가 발생했습니다."
                return state

            state["answer"] = report.get("report", "보고서를 생성하지 못했습니다.")
            logger.info(f"✅ 보고서 생성 완료 (길이: {len(state['answer'])})")

            # 현재 응답에서 생성된 차트/파일 저장
            if report.get("charts"):
                state["current_charts"] = report["charts"]
                logger.info(f"📊 현재 응답 차트 저장: {report['charts']}")
                # analysis_data에도 차트 경로 저장 (후속 PDF 저장 요청을 위해)
                if "analysis_data" in state and isinstance(state["analysis_data"], dict):
                    state["analysis_data"]["charts"] = report["charts"]
            else:
                state["current_charts"] = []  # 차트 생성 안 했으면 빈 리스트

            if report.get("saved_path"):
                state["current_saved_file"] = report["saved_path"]
                logger.info(f"💾 현재 응답 파일 저장: {report['saved_path']}")
            else:
                state["current_saved_file"] = None

            # analysis_data는 다음 후속 질문을 위한 참조용으로 유지
            if report.get("charts") or report.get("saved_path"):
                if "analysis_data" not in state:
                    state["analysis_data"] = {}

                if report.get("charts"):
                    state["analysis_data"]["chart_paths"] = report["charts"]

                if report.get("saved_path"):
                    state["analysis_data"]["saved_file_path"] = report["saved_path"]

        except Exception as e:
            logger.error(f"❌ 보고서 생성 중 오류: {e}", exc_info=True)
            state["answer"] = f"보고서 생성 중 오류가 발생했습니다: {str(e)}"

        return state

    def quality_evaluator_node(self, state: WorkflowState) -> WorkflowState:
        """생성된 답변의 품질을 평가하고 실패 시 재시도 여부를 결정합니다.

        동일한 실패 사유가 2회 이상 반복되거나 총 재시도 횟수가 2회 이상이면 조기 종료하고
        사용자 안내 메시지를 반환합니다. 품질 통과 시 쿼리를 재작성하여 재시도합니다.
        """
        question = state.get("question", "")
        answer = state.get("answer", "")
        result = self.quality_evaluator.evaluate_answer(question, answer)

        state["quality_detail"] = result
        state["quality_passed"] = result.get("status") == "pass"

        if not state["quality_passed"]:
            current_failure = result.get("failure_reason", "unknown")
            previous_failure = state.get("previous_failure_reason", "")

            # 연속 동일 실패 감지
            if current_failure == previous_failure:
                state["consecutive_same_failures"] = state.get("consecutive_same_failures", 0) + 1
            else:
                state["consecutive_same_failures"] = 1

            state["previous_failure_reason"] = current_failure
            state["retries"] = state.get("retries", 0) + 1
            if state['retries'] >= 2:
                logger.warning(
                    f"⚠️ 실패 횟수가 {state['retries']}회 반복됨에도 불구하고 기준 미만 답변생성으로 인하여 조기 종료."
                )
                state['answer'] = (
                    "죄송합니다. 여러 시도에도 만족스러운 답변을 생성하지 못했습니다.\n\n"
                    "질문을 더 구체적으로 작성하시거나, 다른 방식으로 표현해주시면 더 나은 답변을 드릴 수 있습니다."
                )
                state['route'] = 'end'
                return state
                

            # 같은 이유로 2번 이상 실패하면 조기 종료
            if state["consecutive_same_failures"] >= 2:
                logger.warning(
                    f"⚠️ 동일한 실패 사유 ({current_failure})가 {state['consecutive_same_failures']}회 반복됨. 조기 종료."
                )
                state["answer"] = (
                    "죄송합니다. 여러 시도에도 만족스러운 답변을 생성하지 못했습니다.\n\n"
                    "질문을 더 구체적으로 작성하시거나, 다른 방식으로 표현해주시면 더 나은 답변을 드릴 수 있습니다."
                )
                state["route"] = "end"

                if current_failure == "error":
                    state["answer"] = (
                        "죄송합니다. 시스템에서 해당 질문을 처리하는 데 반복적으로 문제가 발생했습니다.\n\n"
                        "다음을 시도해보세요:\n"
                        "1. 질문을 다르게 표현해주세요\n"
                        "2. 더 구체적인 정보를 포함해주세요 (예: 회사명, 날짜 등)\n"
                        "3. 다른 주제로 질문해주세요"
                    )
                else:
                    state["answer"] = (
                        "죄송합니다. 여러 시도에도 만족스러운 답변을 생성하지 못했습니다.\n\n"
                        "질문을 더 구체적으로 작성하시거나, 다른 방식으로 표현해주시면 더 나은 답변을 드릴 수 있습니다."
                    )
                state["route"] = "end"
                return state

            rewrite_result = rewrite_query(
                original_query=question,
                failure_reason=current_failure,
                llm=self.shared_llm,
            )
            logger.info(
                "quality_evaluator_node 결과 | needs_user_input=%s | rewritten_query=%s",
                rewrite_result.get("needs_user_input"),
                rewrite_result.get("rewritten_query"),
            )

            if rewrite_result.get("needs_user_input"):
                state["answer"] = rewrite_result.get(
                    "request_for_detail_msg", "질문을 좀 더 구체적으로 말씀해 주시겠어요? "
                )
                state["route"] = "end"
            else:
                state["question"] = rewrite_result.get("rewritten_query", question)
                state["answer"] = "질문을 다시 정제했습니다. 재시도합니다."
                state["route"] = "retry"  
        else:
            # 성공 시 모든 카운터 초기화
            state["retries"] = 0
            state["consecutive_same_failures"] = 0
            state["previous_failure_reason"] = ""
            state['route'] = 'end'

        return state

    # ------------------------------------------------------------------ #
    # Edge routing helpers
    # ------------------------------------------------------------------ #
    def _route_from_request_analyst(self, state: WorkflowState) -> Literal["end", "supervisor", "report_generator", "general_conversation"]:
        """
        request_analyst에서 다음 노드로 라우팅합니다.
        - 후속 질문(차트/PDF 요청) → report_generator로 직행
        - 금융 질문 → supervisor
        - 일반 대화 → general_conversation
        - 비금융 질문 → end
        """
        route = state.get("route", "supervisor")
        if route == "report_generator":
            logger.info("🎯 request_analyst → report_generator 직행 (후속 질문)")
            return "report_generator"
        elif route == "general_conversation":
            logger.info("💬 request_analyst → general_conversation (일반 대화)")
            return "general_conversation"
        elif route == "end":
            return "end"
        else:
            return "supervisor"

    def _route_from_supervisor(self, state: WorkflowState) -> Literal["financial_analyst", "report_generator", "general_conversation", "end"]:
        return state.get("route", "financial_analyst")

    def _route_from_quality_evaluator(self, state: WorkflowState) -> Literal["retry", "end"]:
        """
        품질 평가 결과에 따라 재시도 여부를 결정합니다.
        최대 3회까지만 재시도하며, 이후에는 강제로 종료합니다.
        """
        route = state.get("route", "end")
        logger.info(f"품질 평가 후 라우팅: {route}")
        return route  

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        question: str,
        previous_messages: list = None,
        previous_analysis_data: dict = None,
        session_id: str = None
    ) -> WorkflowState:
        """사용자 질문에 따른 그래프를 실행한 뒤 최종 상태를 반환합니다."""
        # 질문 시작 구분선
        logger.info("=" * 80)
        logger.info(f"🔵 새로운 질문 처리 시작: {question[:50]}..." if len(question) > 50 else f"🔵 새로운 질문 처리 시작: {question}")
        logger.info("=" * 80)

        # State 초기화 - 모든 필드를 명시적으로 초기화
        initial_state: WorkflowState = {
            "question": question,
            "answer": "",
            "route": "",
            "retries": 0,  # 재시도 카운터 초기화
            "quality_passed": False,
            "rag_search_results": [],
            "consecutive_same_failures": 0,  # 연속 실패 카운터 초기화
            "previous_failure_reason": "",  # 이전 실패 이유 초기화
            "messages": previous_messages if previous_messages is not None else []
        }

        # 이전 분석 데이터가 있으면 state에 추가 (후속 질문 감지용)
        if previous_analysis_data is not None:
            initial_state["analysis_data"] = previous_analysis_data
            logger.info(f"✅ 이전 분석 데이터 로드 완료 - type: {previous_analysis_data.get('analysis_type', 'N/A')}")

        result = self.graph.invoke(initial_state)

        # 질문 종료 구분선
        logger.info("=" * 80)
        logger.info(f"🟢 질문 처리 완료 - route: {result.get('route')}, quality_passed: {result.get('quality_passed')}, retries: {result.get('retries', 0)}")
        logger.info("=" * 80)
        logger.info("")  # 빈 줄 추가

        return result


def build_workflow() -> Workflow:
    """외부에서 간편하게 워크플로우 인스턴스를 생성할 때 사용."""
    return Workflow()


__all__ = ["Workflow", "WorkflowState", "build_workflow"]


# if __name__ == "__main__":
#     # from IPython.display import Image
#     wf = build_workflow()
#     # Image(wf.graph.get_graph().draw_png())
#     mermaid_code = wf.graph.get_graph().draw_mermaid()
#     # print(wf.graph.get_graph().draw_mermaid())
#     with open("/workspace/langchain_project/img/workflow_diagram.mmd", "w", encoding="utf-8") as f:
#         f.write(mermaid_code)
#     print("워크플로우 다이어그램이 workflow_diagram.mmd 파일로 저장되었습니다.")


if __name__ == "__main__":
    workflow = build_workflow()
    sample_questions = [
        "삼성전자와 애플의 최근 실적을 비교해주세요.",
        "애플 주식 분석 보고서를 차트와 함께 PDF로 저장해줘. 파일명은 너가 생각해서 적절한 걸 정해줘.",
        "내일 날씨는 뭐야?",
        "레버리지 ETF의 위험성을 설명해줘",
        "테슬라 주식을 분석해서 마크다운 파일로 저장해줘. 적절한 파일명으로.",
        "삼성전자와 애플의 최근 주가를 비교 후, 간단하게 차트를 그리고 pdf파일로 저장해 줘.",
        "나스닥이 뭐야?",
        "모바일로 주식 거래하는 앱은 뭐라고 해?"
    ]

    for question in sample_questions:
        print("=" * 80)
        print(f"Q: {question}")
        result = workflow.run(question)
        print(f"route: {result.get('route')}")
        answer = result.get("answer")
        if isinstance(answer, str) and len(answer) > 400:
            answer = answer[:400] + "..."
        print("answer:", answer)
        if result.get("rag_search_results"):
            print("rag_search_results:")
            for line in result["rag_search_results"]:
                print(f"  {line}")
        if result.get("quality_detail"):
            print(f"quality_check: {result['quality_detail']}")
