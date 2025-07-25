"""
YAJPH: The Anti-Black-Box Engine
GitHub's First Explainable Decision Framework

Every "no" comes with a roadmap to "yes".
"""

import yaml
import json
import os
import subprocess
import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Decision:
    """The explainable decision result"""
    passed: bool
    score: Optional[float] = None
    missing: Dict[str, str] = None
    rejected_on: Optional[str] = None
    your_score: Optional[Any] = None
    threshold: Optional[Any] = None
    fix: Optional[str] = None
    resources: List[str] = None
    audit_trail: List[str] = None
    
    def __post_init__(self):
        if self.missing is None:
            self.missing = {}
        if self.resources is None:
            self.resources = []
        if self.audit_trail is None:
            self.audit_trail = []


@dataclass
class TrialResult:
    """Result of an epistemic trial"""
    status: str  # "approved", "rejected", "pending_human_review"
    confidence: float
    claim: str
    code_diff: str
    agent_outputs: Dict[str, Any]
    deterministic_checks: Dict[str, Any]
    required_changes: List[str]
    simulation_results: Dict[str, Any]
    timestamp: str
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


class AgentOrchestrator:
    """Handles dynamic agent loading and execution"""
    
    def __init__(self, agents_config: Dict[str, Any]):
        self.agents_config = agents_config
        self.agent_outputs = {}
    
    def execute_agent(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent with given context"""
        if agent_name not in self.agents_config:
            raise ValueError(f"Agent '{agent_name}' not found in configuration")
        
        agent_config = self.agents_config[agent_name]
        
        # For now, this is a placeholder that simulates agent execution
        # In a real implementation, this would call actual LLM APIs
        result = {
            "agent": agent_name,
            "model": agent_config.get("model", "placeholder"),
            "prompt_template": agent_config.get("prompt_template", ""),
            "weight": agent_config.get("weight", 1.0),
            "output": f"Simulated output from {agent_name} for claim: {context.get('claim', 'N/A')}",
            "confidence": agent_config.get("default_confidence", 0.8),
            "issues": [],
            "suggestions": []
        }
        
        self.agent_outputs[agent_name] = result
        return result
    
    def execute_agent_chain(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain of agents, passing outputs between them"""
        chain_results = {}
        current_context = context.copy()
        
        for agent_name in agents:
            # Add previous agent outputs to context
            current_context["previous_agents"] = chain_results
            
            result = self.execute_agent(agent_name, current_context)
            chain_results[agent_name] = result
            
            # Update context with current agent's output for next agent
            current_context[f"{agent_name}_output"] = result
        
        return chain_results


class DeterministicChecker:
    """Handles deterministic code checks that don't require LLMs"""
    
    def __init__(self, checks_config: Dict[str, Any]):
        self.checks_config = checks_config
    
    def run_check(self, check_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single deterministic check"""
        if check_name not in self.checks_config:
            raise ValueError(f"Check '{check_name}' not found in configuration")
        
        check_config = self.checks_config[check_name]
        check_type = check_config.get("type", "script")
        
        if check_type == "script":
            return self._run_script_check(check_name, check_config, context)
        elif check_type == "function":
            return self._run_function_check(check_name, check_config, context)
        else:
            raise ValueError(f"Unknown check type: {check_type}")
    
    def _run_script_check(self, check_name: str, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a shell script check"""
        script = config.get("script", "")
        expected_exit_code = config.get("expected_exit_code", 0)
        
        try:
            result = subprocess.run(script, shell=True, capture_output=True, text=True, timeout=30)
            passed = result.returncode == expected_exit_code
            
            return {
                "check": check_name,
                "type": "script",
                "passed": passed,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "script": script
            }
        except subprocess.TimeoutExpired:
            return {
                "check": check_name,
                "type": "script",
                "passed": False,
                "error": "Script execution timed out",
                "script": script
            }
        except Exception as e:
            return {
                "check": check_name,
                "type": "script",
                "passed": False,
                "error": str(e),
                "script": script
            }
    
    def _run_function_check(self, check_name: str, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a Python function check"""
        # This would implement custom Python function execution
        # For now, return a placeholder
        return {
            "check": check_name,
            "type": "function",
            "passed": True,
            "message": f"Function check {check_name} placeholder"
        }
    
    def run_all_checks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all configured deterministic checks"""
        results = {}
        
        for check_name in self.checks_config:
            try:
                results[check_name] = self.run_check(check_name, context)
            except Exception as e:
                results[check_name] = {
                    "check": check_name,
                    "passed": False,
                    "error": f"Check execution failed: {str(e)}"
                }
        
        return results


class Router:
    """The core YAJPH reasoning engine"""
    
    def __init__(self, input_schema: str = None, output_schema: str = None, 
                 rules: Dict[str, Any] = None, models: Dict[str, str] = None):
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.rules = rules or {}
        self.models = models or {}
        
        # Load rules from YAML if provided
        if input_schema and input_schema.endswith('.yaml'):
            with open(input_schema, 'r') as f:
                self.rules = yaml.safe_load(f)
    
    def evaluate(self, data: Dict[str, Any]) -> Decision:
        """
        The magic happens here: transparent, explainable decisions
        """
        audit_trail = [f"evaluating: {list(data.keys())}"]
        
        # Check each rule
        for rule_name, threshold in self.rules.get('requirements', {}).items():
            audit_trail.append(f"checking: {rule_name}")
            
            if rule_name not in data:
                return Decision(
                    passed=False,
                    missing={rule_name: f"required but not provided"},
                    rejected_on=rule_name,
                    audit_trail=audit_trail,
                    fix=f"Please provide {rule_name} information"
                )
            
            user_value = data[rule_name]
            
            # Handle different comparison types
            if isinstance(threshold, str) and threshold.endswith('%'):
                # DTI percentage check
                threshold_val = float(threshold.rstrip('%'))
                user_val = float(str(user_value).rstrip('%'))
                
                if user_val > threshold_val:
                    return Decision(
                        passed=False,
                        rejected_on=rule_name,
                        your_score=f"{user_val}%",
                        threshold=threshold,
                        audit_trail=audit_trail,
                        fix=f"Reduce {rule_name} to below {threshold}"
                    )
            
            elif isinstance(threshold, (int, float)):
                # Numeric threshold (like credit score or skill level)
                if isinstance(user_value, str) and '/' in user_value:
                    # Handle "2/5" format
                    user_val = float(user_value.split('/')[0])
                else:
                    user_val = float(user_value)
                
                if user_val < threshold:
                    return Decision(
                        passed=False,
                        rejected_on=rule_name,
                        your_score=user_value,
                        threshold=threshold,
                        missing={rule_name: f"{user_value} (need {threshold})"},
                        audit_trail=audit_trail,
                        resources=self._get_resources(rule_name),
                        fix=f"Improve {rule_name} to at least {threshold}"
                    )
        
        # Check required items
        for required_item in self.rules.get('must_have', []):
            if required_item not in data or not data[required_item]:
                return Decision(
                    passed=False,
                    rejected_on=required_item,
                    missing={required_item: "required but missing"},
                    audit_trail=audit_trail,
                    fix=f"Please provide {required_item}"
                )
        
        # All checks passed!
        audit_trail.append("all_checks_passed")
        return Decision(
            passed=True,
            audit_trail=audit_trail,
            fix="No action needed - you qualify!"
        )
    
    def trial(self, claim: str, code_diff: str, config_file: str = "code_review.yaml") -> TrialResult:
        """
        Execute an epistemic trial for code changes
        """
        # Load the epistemic configuration
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        context = {
            "claim": claim,
            "code_diff": code_diff,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Initialize components
        agent_orchestrator = AgentOrchestrator(config.get("agents", {}))
        deterministic_checker = DeterministicChecker(config.get("deterministic_checks", {}))
        
        # Run deterministic checks first
        deterministic_results = deterministic_checker.run_all_checks(context)
        
        # Execute agent chain
        agent_chain = config.get("agent_chain", [])
        agent_results = agent_orchestrator.execute_agent_chain(agent_chain, context)
        
        # Calculate overall confidence and make decision
        confidence = self._calculate_confidence(agent_results, deterministic_results, config)
        status = self._determine_status(confidence, deterministic_results, config)
        
        # Generate required changes if any
        required_changes = self._extract_required_changes(agent_results, deterministic_results)
        
        # Create trial result
        result = TrialResult(
            status=status,
            confidence=confidence,
            claim=claim,
            code_diff=code_diff,
            agent_outputs=agent_results,
            deterministic_checks=deterministic_results,
            required_changes=required_changes,
            simulation_results={},  # Placeholder for future simulation integration
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # Save trial log and verdict
        self._save_trial_outputs(result, config)
        
        return result
    
    def _calculate_confidence(self, agent_results: Dict[str, Any], 
                            deterministic_results: Dict[str, Any], 
                            config: Dict[str, Any]) -> float:
        """Calculate overall confidence based on agent and deterministic results"""
        total_weight = 0
        weighted_confidence = 0
        
        # Factor in agent confidences with their weights
        for agent_name, result in agent_results.items():
            weight = result.get("weight", 1.0)
            confidence = result.get("confidence", 0.5)
            
            weighted_confidence += weight * confidence
            total_weight += weight
        
        # Factor in deterministic check results
        deterministic_weight = config.get("deterministic_weight", 2.0)
        deterministic_pass_rate = sum(1 for r in deterministic_results.values() if r.get("passed", False)) / max(len(deterministic_results), 1)
        
        weighted_confidence += deterministic_weight * deterministic_pass_rate
        total_weight += deterministic_weight
        
        return weighted_confidence / max(total_weight, 1)
    
    def _determine_status(self, confidence: float, deterministic_results: Dict[str, Any], 
                         config: Dict[str, Any]) -> str:
        """Determine the trial status based on confidence and results"""
        # Check if any deterministic checks failed critically
        for result in deterministic_results.values():
            if not result.get("passed", False) and result.get("critical", False):
                return "rejected"
        
        # Use confidence thresholds
        approval_threshold = config.get("approval_threshold", 0.8)
        rejection_threshold = config.get("rejection_threshold", 0.3)
        
        if confidence >= approval_threshold:
            return "approved"
        elif confidence <= rejection_threshold:
            return "rejected"
        else:
            return "pending_human_review"
    
    def _extract_required_changes(self, agent_results: Dict[str, Any], 
                                deterministic_results: Dict[str, Any]) -> List[str]:
        """Extract required changes from agent and deterministic results"""
        changes = []
        
        # Extract from agent suggestions
        for agent_name, result in agent_results.items():
            suggestions = result.get("suggestions", [])
            changes.extend(suggestions)
        
        # Extract from failed deterministic checks
        for check_name, result in deterministic_results.items():
            if not result.get("passed", False):
                error_msg = result.get("error", f"Check '{check_name}' failed")
                changes.append(f"Fix: {error_msg}")
        
        return changes
    
    def _save_trial_outputs(self, result: TrialResult, config: Dict[str, Any]):
        """Save TRIAL.log and VERDICT.json files"""
        output_dir = config.get("output_dir", ".")
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save TRIAL.log
        trial_log_path = os.path.join(output_dir, "TRIAL.log")
        with open(trial_log_path, 'w') as f:
            f.write(f"# Epistemic Trial Log\n")
            f.write(f"**Timestamp:** {result.timestamp}\n")
            f.write(f"**Claim:** {result.claim}\n")
            f.write(f"**Status:** {result.status}\n")
            f.write(f"**Confidence:** {result.confidence:.2f}\n\n")
            
            f.write(f"## Code Changes\n```diff\n{result.code_diff}\n```\n\n")
            
            f.write(f"## Deterministic Checks\n")
            for check_name, check_result in result.deterministic_checks.items():
                status = "✅ PASS" if check_result.get("passed") else "❌ FAIL"
                f.write(f"- **{check_name}:** {status}\n")
                if not check_result.get("passed"):
                    f.write(f"  - Error: {check_result.get('error', 'N/A')}\n")
            f.write("\n")
            
            f.write(f"## Agent Analysis\n")
            for agent_name, agent_result in result.agent_outputs.items():
                f.write(f"### {agent_name}\n")
                f.write(f"- **Model:** {agent_result.get('model')}\n")
                f.write(f"- **Confidence:** {agent_result.get('confidence'):.2f}\n")
                f.write(f"- **Output:** {agent_result.get('output')}\n\n")
            
            if result.required_changes:
                f.write(f"## Required Changes\n")
                for change in result.required_changes:
                    f.write(f"- {change}\n")
        
        # Save VERDICT.json
        verdict_path = os.path.join(output_dir, "VERDICT.json")
        verdict_data = {
            "status": result.status,
            "confidence": result.confidence,
            "timestamp": result.timestamp,
            "required_changes": result.required_changes,
            "summary": {
                "deterministic_checks_passed": sum(1 for r in result.deterministic_checks.values() if r.get("passed", False)),
                "total_deterministic_checks": len(result.deterministic_checks),
                "agents_consulted": list(result.agent_outputs.keys())
            }
        }
        
        with open(verdict_path, 'w') as f:
            json.dump(verdict_data, f, indent=2)
    
    def _get_resources(self, skill: str) -> List[str]:
        """Suggest helpful resources for improvement"""
        resources_map = {
            'sql': ['https://sqlzoo.net', 'https://w3schools.com/sql'],
            'python': ['https://python.org/tutorial', 'https://codecademy.com/python'],
            'credit_score': ['https://creditkarma.com', 'https://annualcreditreport.com']
        }
        return resources_map.get(skill.lower(), [])


def attach_router(input_schema: str, output_schema: str, 
                 llm_router=None) -> Router:
    """
    Attach YAJPH to your existing application
    """
    return Router(input_schema=input_schema, output_schema=output_schema)


def evaluate_yaml(rules_file: str, data_file: str) -> Dict[str, Any]:
    """
    CLI-friendly function for YAML-to-JSON evaluation
    """
    # Load rules
    with open(rules_file, 'r') as f:
        rules = yaml.safe_load(f)
    
    # Load data
    with open(data_file, 'r') as f:
        if data_file.endswith('.yaml') or data_file.endswith('.yml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    # Create router and evaluate
    router = Router(rules=rules)
    result = router.evaluate(data)
    
    return asdict(result)


def run_trial(claim: str, code_diff: str, config_file: str = "code_review.yaml") -> Dict[str, Any]:
    """
    CLI-friendly function for running epistemic trials
    """
    router = Router()
    result = router.trial(claim, code_diff, config_file)
    return asdict(result)


def deploy(input_schema: str, output_schema: str, api_mode: str = "rest", port: int = 8080):
    """
    Deploy YAJPH as a web service (placeholder for now)
    """
    print(f"🚀 YAJPH would deploy on port {port}")
    print(f"📄 Input schema: {input_schema}")
    print(f"📄 Output schema: {output_schema}")
    print(f"🌐 Mode: {api_mode}")
    print("(Full deployment coming in v0.2)")


# CLI entry point
def main():
    """Enhanced CLI with trial command support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YAJPH: The Anti-Black-Box Engine")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Original evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate data against rules')
    eval_parser.add_argument('--rules', required=True, help='YAML rules file')
    eval_parser.add_argument('--input', required=True, help='Input data file')
    eval_parser.add_argument('--output', help='Output JSON file (optional)')
    
    # New trial command
    trial_parser = subparsers.add_parser('trial', help='Run epistemic trial for code changes')
    trial_parser.add_argument('--claim', required=True, help='The claim/commit message')
    trial_parser.add_argument('--code-diff', help='Code diff (or will read from git)')
    trial_parser.add_argument('--config', default='code_review.yaml', help='Trial configuration file')
    trial_parser.add_argument('--output', help='Output directory for trial results')
    
    args = parser.parse_args()
    
    if args.command == 'evaluate':
        result = evaluate_yaml(args.rules, args.input)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
    
    elif args.command == 'trial':
        # Get code diff from git if not provided
        code_diff = args.code_diff
        if not code_diff:
            try:
                result = subprocess.run(['git', 'diff', 'HEAD^', 'HEAD'], 
                                      capture_output=True, text=True)
                code_diff = result.stdout
            except Exception:
                code_diff = "No diff available"
        
        # Run the trial
        try:
            result = run_trial(args.claim, code_diff, args.config)
            
            if args.output:
                output_path = os.path.join(args.output, 'trial_result.json')
                os.makedirs(args.output, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                print(json.dumps(result, indent=2))
                
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Make sure '{args.config}' exists in the current directory")
        except Exception as e:
            print(f"Trial execution failed: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
