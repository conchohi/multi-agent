"""
계획에 기반하여 하위 에이전트에게 일 수행
"""
from typing import List, Dict, Any

from app.state import AgentState, RoutingDecision
from app.model.settings import AgentConfig
from app.util.logger import get_logger

logger = get_logger(__name__)

class Supervisor:
    def __init__(self, agent_configs: List[AgentConfig]):
        self.agent_configs = agent_configs
    
    async def supervisor_node(self, state: AgentState) -> Dict[str, Any]:
        """
        에이전트 라우팅

        Plan에 따라 다음 에이전트를 선택합니다.

        Args:
            state: SupervisorState

        Returns:
            라우팅 결정을 포함한 부분 State
        """
        execution_plan = state.get('plan')
        
        steps = execution_plan.steps
        execution_mode = execution_plan.execution_mode
        
        if execution_mode == 'sequential':
            current_step = state.get('current_step', 0)
            total_steps = execution_plan.total_steps
            
            if current_step >= total_steps:
                logger.info('[SUPERVISOR] 순차 실행 단계 종료 -> EVALUATOR 노드 이동')
                return {
                    "routing_decision" : RoutingDecision()
                }
            next_step = steps[current_step]
            
            logger.info(f'[SUPERVISOR] 순차 실행 {next_step.step_number} 단계 -> [{next_step.agent}] 노드 이동')
            return {
                "routing_decision" : RoutingDecision(
                    next_step=next_step
                ),
                "current_step" : current_step + 1
            }
        else:  # parallel execution
            running_steps = state.get('running_steps', False)
            
            if not running_steps:
                logger.info(
                    f'[SUPERVISOR] 병렬 실행 {len(steps)}개 step 동시 실행: '
                    f'{[(s.step_number, s.agent, s.task) for s in steps]}'
                )

                return {
                    "routing_decision": RoutingDecision(
                        next_steps=steps,
                        is_parallel=True
                    ),
                    "running_steps" : True
                }
            else :
                logger.info('[SUPERVISOR] 병렬 실행 단계 종료 -> EVALUATOR 노드 이동')
            return {
                "routing_decision" : RoutingDecision(),
                "running_steps" : False
            }


            
            
            
            
            
            