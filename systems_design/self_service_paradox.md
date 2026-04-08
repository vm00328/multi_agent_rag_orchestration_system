# 1. Problem Statement

The platform must support two very different user groups without creating two separate systems:

- **Business Users** need fast time-to-value, low cognitive load, and guardrails.
- **Power Users** need flexibility, custom logic, direct integrations, and programmatic control.
- **Platform Engineering** needs one architecture that is secure, maintainable, auditable, and scalable.

The design challenge is to make the platform easy enough for non-technical users and powerful enough for technical users, while avoiding fragmented tooling, duplicated runtimes, or inconsistent governance.

---

# 2. Proposed Solution

The proposed solution is a single governed automation platform where no-code, low-code, and pro-code experiences all compile into the same canonical workflow model and run on the same secure execution engine, with progressive disclosure controlling how much power each user sees.

This means we do not design two products for two user groups. We design one platform with one execution model, then expose that model through different levels of abstraction.

---

# 3. Design Principles

## 3.1 One Platform, One Runtime

All workflows should execute on the same orchestration engine, regardless of whether they were created via templates, visual editing, or code.

## 3.2 Progressive Disclosure

The platform should expose capabilities in layers. Beginners start with safe abstractions. Advanced users can unlock lower-level control without switching products.

## 3.3 Safe Extensibility

Power is introduced through governed extension points: custom blocks, connectors, packaged logic, and APIs that still run inside the same policy and observability model.

## 3.4 Policy by Default

Auditability, approvals, Role based access control (RBAC), secret management, data classification, and environment separation must be built into the platform rather than added on later.

---

# 4. Architecture Design

## 4.1 Proposed Architecture

The paradox centers on finding a middle ground between simplicity and power. This can be visualized as a layered abstraction model. The platform consists of four major layers:

- **Layer 1 - Foundation:** A robust set of microservices and APIs that perform core banking and AI functions.
- **Layer 2 - Orchestration and Logic:** A shared engine that translates user intent, whether expressed via code or blocks, into executable workflows.
- **Layer 3 - Experience Layer:** A single platform with multiple authoring experiences over the same workflow model.
  - Business UI: templates, forms, wizard, drag-and-drop builder
  - Power UI: code editor, API access, SDK/CLI, advanced debugger
- **Layer 4 - Governance:** Role-based access control, audit logs, approval gates, sandbox and production environments, and observability.

## 4.2 Core Architectural Decisions

The critical decision is that all authoring modes compile to the same canonical workflow definition and execute on the same runtime. Both user types build against the same composition and execution layers, just through different interfaces.

To satisfy the requirement for a unified platform, the architecture should be centered on a **Shared Representation Model**.

### Unified Domain Specific Language (DSL)

The platform should use a canonical JSON-based workflow schema to define workflows. For example, when a business user drags a "Send Email" block, it generates a structured workflow definition. When a power user writes code, it compiles to that same canonical format.

### Progressive Disclosure Strategy

This is both a UX and architectural principle where only the most relevant features are shown initially.

- **No-Code Mode:** Drag-and-drop blocks for standard banking microservices
- **Low-Code Mode:** Property panels where users can add simple logic or data mappings
- **Pro-Code Mode:** A "View Source" or "Edit Code" path that allows power users to add governed custom logic through packaged extensions or sandboxed functions

### Interface Connections

A React/TypeScript frontend can provide a dynamic canvas that switches between visual and code views seamlessly, while preserving the same underlying workflow representation.

## 4.3 High-Level System Diagram
```
+---------------------------------------------------------------+
|                        Experience Layer                       |
|                                                               |
|  [Templates] [Wizard] [Visual Builder] [Advanced Editor/API]  |
+---------------------------+-----------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|                    Composition / Logic Layer                  |
|                                                               |
|  Canonical Workflow Model | Validation | Policy Checks        |
|  Versioning | Packaging | Shared DSL                          |
+---------------------------+-----------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|                        Execution Layer                        |
|                                                               |
|  Orchestrator | Connector Runtime | Job Runner                |
|  State Store  | Retries | Rollbacks | Scheduling              |
+---------------------------+-----------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|                  Governance & Platform Services               |
|                                                               |
| RBAC | Audit | Secrets | Approvals | Monitoring | Environments|
+---------------------------------------------------------------+
```
---

# 5. User Experience Strategy

## 5.1 Business User Experience

The business experience should optimize for confidence and speed. The plan for business users is as follows:

- Start from business-oriented templates
- Use plain-language labels instead of technical ones
- Show only approved systems and actions
- Use guided onboarding
- Visualize workflow outcomes, dependencies, and approvals
- Prevent invalid configurations at design time

## 5.2 Power User Experience

The power-user experience should optimize for flexibility and precision:

- Access to code editor and schema view
- Full workflow diff and version history
- API access and CI/CD integration
- Custom connector and reusable module creation
- Advanced debugging and execution tracing

## 5.3 Transitions Between Capabilities

Transitions are critical. Users should not switch tools. They should open the same workflow in a more advanced mode.


```text
                         Single Unified Platform
┌──────────────────────────────────────────────────────────────────────┐
│                 Shared Workflow Model + Shared Runtime               │
└──────────────────────────────────────────────────────────────────────┘

Business User Entry                                              Power User Entry
        │                                                               │
        v                                                               v
┌──────────────────┐                                           ┌──────────────────┐
│ Template Gallery │                                           │ Code / API / SDK │
└─────────┬────────┘                                           └─────────┬────────┘
          v                                                              v
┌──────────────────┐                                           ┌──────────────────┐
│ Guided Wizard    │                                           │ Custom Logic /   │
│ + Forms          │                                           │ Connectors       │
└─────────┬────────┘                                           └─────────┬────────┘
          v                                                              v
┌──────────────────┐                                           ┌──────────────────┐
│ Visual Builder   │<──────────── Same Workflow ────────────> │ Schema / Advanced │
│                  │              Definition                  │ Editor            │
└─────────┬────────┘                                           └─────────┬────────┘
          └──────────────────────────────┬───────────────────────────────┘
                                         v
                         ┌────────────────────────────────┐
                         │ Validation + Policy Checks     │
                         └────────────────┬───────────────┘
                                          v
                         ┌────────────────────────────────┐
                         │ Secure Execution Engine        │
                         └────────────────┬───────────────┘
                                          v
                         ┌────────────────────────────────┐
                         │ Audit, Monitoring, Approvals   │
                         └────────────────────────────────┘
```
## 5.4 Onboarding Pathways

The onboarding model should be persona-based, not platform-based:

- **Business path**: templates -> wizard -> visual builder
- **Technical path**: sample repos -> SDK -> APIs -> custom modules
- **Shared path**: governance, security, approvals, and production readiness

This ensures all users understand the platform’s operating model even if they use it differently.

---

# 6. Technical Implementation
## 6.1 Core Components

The platform should include the following key components:

- Workflow Definition Service for storing, validating, and versioning workflow definitions
- Policy Engine for enforcing compliance, data, and operational rules
- Execution Orchestrator for running workflows, managing dependencies, and handling retries
- Connector Framework for reusable integrations with internal and external systems
- Extension SDK for controlled creation of reusable custom steps, connectors, and validators
- Observability Stack for metrics, logs, traces, and failure analysis

## 6.2 Suggested Technology Choices

- **Frontend**: React / TypeScript for the authoring experience
- **Backend APIs**: Python services, for example FastAPI, for orchestration and platform services
- **Database**: PostgreSQL for workflow definitions, version history, execution metadata, and audit records
- **Execution Layer**: Worker services that parse the canonical workflow definition and execute steps in order

## 6.3 Extension Mechanisms

Power users should be able to extend the platform in controlled ways:

- Custom Connectors for integrating new systems
- Custom Logic Nodes for specialized processing
- Reusable Subflows / Packs that convert advanced logic into business-consumable components

Every extension should:

- declare inputs and outputs
- define required permissions
- specify data sensitivity
- support testing
- emit telemetry
- follow packaging and versioning rules

---

# 7. Governance, Security, and Compliance

Controlled automation is critical to the success of the platform. The following controls should be embedded into the system architecture.

## 7.1 Role-Based Access Control

Different permissions should apply to:

- template consumers
- workflow authors
- extension developers
- approvers
- platform admins

## 7.2 Environment Separation

Workflows must move through controlled environments such as:

- sandbox
- integration
- user acceptance testing
- production

Promotion should require validation and, for sensitive workflows, formal approval.

## 7.3 Auditability

Every important action should be recorded:

- who created or modified a workflow
- what changed
- who approved it
- what systems it touched
- what data classification applied
- which version ran in production

## 7.4 Guardrails and Platform-Wide Constraints

The platform should embed the following controls as default guardrails and non-negotiable operating constraints.

**Guardrails**
- approval gates for high-risk actions
- restricted connectors for regulated systems
- mandatory logging for critical actions
- secret rotation and scoped credential usage

**Platform-wide constraints**
- maker-checker approvals for risky workflows
- environment separation across sandbox, UAT, and production
- role-based permissions for actions and connectors
- immutable audit trails
- versioned deployments and rollback
- data residency and classification enforcement
- observability and incident tracing

```text
[Design Workflow]
       |
       v
[Validate Schema and Policies]
       |
       +---- fail ----> [Return Errors to Author]
       |
       v
[Sandbox Test Execution]
       |
       +---- fail ----> [Fix and Re-test]
       |
       v
[Approval Gate]
       |
       +---- rejected ---> [Revise Workflow]
       |
       v
[Promote to Production]
       |
       v
[Execute Workflow]
       |
       v
[Observe Runs]
  - logs
  - metrics
  - traces
  - audit trail
       |
       +---- incident ----> [Rollback / Disable / Investigate]
       |
       v
[Iterate and Version Next Release]
```

---

# 8. Long-Term Maintainability

## 8.1 Platform Evolution Strategy

The platform should evolve by extending shared abstractions, not by adding parallel tools.

Examples:

- add new connectors without changing the runtime model
- introduce new visual blocks by packaging existing technical components
- expose advanced features gradually via progressive disclosure

## 8.2 Versioning and Deprecation

Long-term maintainability requires explicit lifecycle management:

- versioned workflow definitions
- versioned connectors and extensions
- compatibility windows
- deprecation notices
- migration tooling

## 8.3 Telemetry-Driven Improvement

Product direction should be guided by usage and failure data:

- which templates are most reused
- which connectors fail most often
- where power users repeatedly create custom logic

This helps prioritize platform investment and reduce repeated complexity.

---

# 9. Trade-offs and Risks

## 9.1 Trade-off: Simplicity vs Flexibility

A unified platform will never feel as simple as a pure no-code tool or as unconstrained as a raw code framework.

Mitigation: progressive disclosure and role-based access keep the experience appropriate to each user.

## 9.2 Trade-off: Governance vs Speed

Approval processes and policy checks can slow delivery.

Mitigation: pre-approved templates

## 9.3 Trade-off: Extensibility vs Platform Stability

Allowing custom code or connectors can create support and security risk.

Mitigation: sandboxing, packaging standards, test requirements, and controlled publication to the shared catalog.

## 9.4 Risk: Over-Abstracting for Business Users

Too much simplification can hide important operational consequences.

Mitigation: provide explainability views, impact summaries, and approval visibility even in simplified interfaces.

# 10. Platform Evolution and AI-Assisted Authoring

This platform can later support AI-assisted workflow authoring.

Examples include:

- business users describing a workflow in natural language
- the system proposing an initial draft workflow
- explaining failures and suggesting improvements

However, AI should remain an assistive layer rather than the core control mechanism. Any AI-generated workflow should still compile into the same governed workflow model and pass the same validation, approval, and execution controls as any manually created workflow.