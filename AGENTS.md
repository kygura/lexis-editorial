# Agents Configuration

This file configures how Claude Code dispatches subagents for parallel task execution.

## Subagent Types

When multiple independent tasks need to be worked on in parallel, Claude Code can spawn specialized subagents:

- **Explore**: Fast exploration of codebases—searching for patterns, file structures, and code locations
- **Plan**: Architecture and implementation planning—designing approaches before coding
- **code-reviewer**: Code review against standards, security, logic, and adherence to patterns
- **feature-dev**: Complete feature development with architecture understanding
- **general-purpose**: Research and multi-step tasks without specialized focus

## Guidelines

1. **Use for independent tasks**: Only dispatch subagents when tasks have no shared state or sequential dependencies
2. **Keep context**: Each subagent gets the full conversation context—no need to repeat information
3. **Parallel execution**: Subagents run simultaneously; results are collected and summarized back to main
4. **One task per agent**: Assign each independent unit of work to a separate subagent

## Example Usage

When you encounter 3+ independent tasks (e.g., "fix bug in auth", "refactor utils", "add tests"), consider:

```
Task 1 (independent): Bug fix in auth module
Task 2 (independent): Refactor utilities  
Task 3 (independent): Add test coverage

→ Dispatch 3 subagents in parallel with Skill: superpowers:subagent-driven-development
```

## Project-Specific Guidance

- **Frontend work**: Use `frontend-design` agent when creating UI components
- **Architecture**: Use `Plan` agent for design decisions before implementation
- **Code quality**: Use `code-reviewer` for validation after implementation
