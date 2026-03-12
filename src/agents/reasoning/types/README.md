
# Reasoning Types System

## Overview
This system provides a flexible framework for reasoning that combines multiple reasoning methodologies. It supports:
- 6 core reasoning types (abduction, deduction, induction, analogical, decompositional, cause-effect)
- Dynamic combination of up to 3 reasoning types
- Context passing between reasoning steps
- Memory-enhanced reasoning with prioritized experience replay

## Core Components

### Reasoning Types
```mermaid
classDiagram
    class BaseReasoning {
        <<Abstract>>
        +perform_reasoning(input, context)
    }
    
    class ReasoningAbduction {
        +perform_reasoning(observations, context)
    }
    
    class ReasoningDeductive {
        +perform_reasoning(premises, hypothesis, context)
    }
    
    class ReasoningInductive {
        +perform_reasoning(observations, context)
    }
    
    class ReasoningAnalogical {
        +perform_reasoning(target, source_domain, context)
    }
    
    class ReasoningDecompositional {
        +perform_reasoning(system, context)
    }
    
    class ReasoningCauseAndEffect {
        +perform_reasoning(events, conditions, context)
    }
    
    BaseReasoning <|-- ReasoningAbduction
    BaseReasoning <|-- ReasoningDeductive
    BaseReasoning <|-- ReasoningInductive
    BaseReasoning <|-- ReasoningAnalogical
    BaseReasoning <|-- ReasoningDecompositional
    BaseReasoning <|-- ReasoningCauseAndEffect
```

### ReasoningTypes -> Manager


```mermaid
classDiagram
    class ReasoningTypes {
        -_task_types: Dict[str, Type[BaseReasoning]]
        -memory: ReasoningMemory
        +discover_task_types()
        +create(task_type: str) BaseReasoning
        -_get_reasoning_class(task_type)
        -_create_combined_reasoning(combined_type)
        -_create_combined_class(reasoning_classes)
    }
    
    class CombinedReasoning {
        -components: List[BaseReasoning]
        +perform_reasoning(input, context)
        -_synthesize_result(results)
    }
    
    ReasoningTypes --> BaseReasoning: Creates
    ReasoningTypes --> CombinedReasoning: Creates
    CombinedReasoning --> BaseReasoning: Composes
```

### ReasoningMemory

```mermaid
classDiagram
    class ReasoningMemory {
        -tree: SumTree
        -tag_index: defaultdict
        -lock: Lock
        +add(experience, priority, tag)
        +sample_proportional(batch_size)
        +save_checkpoint(name)
        +load_checkpoint(path)
        +get_current_context()
    }
    
    class SumTree {
        -capacity: int
        -tree: ndarray
        -data: ndarray
        +add(priority, data)
        +update(data_idx, priority)
        +sample(value)
    }
    
    ReasoningMemory --> SumTree: Uses
```

## System Flow

### Single Reasoning Flow

```mermaid
sequenceDiagram
    participant User
    participant ReasoningTypes
    participant ReasoningInstance
    participant ReasoningMemory
    
    User->>ReasoningTypes: create("abduction")
    ReasoningTypes->>ReasoningInstance: Instantiate ReasoningAbduction
    User->>ReasoningInstance: perform_reasoning(observations, context)
    ReasoningInstance->>ReasoningMemory: Query context
    ReasoningMemory-->>ReasoningInstance: Return context tags
    ReasoningInstance->>ReasoningInstance: Execute reasoning logic
    ReasoningInstance->>ReasoningMemory: Store results
    ReasoningInstance-->>User: Return reasoning result
```

### Combined Reasoning Flow

```mermaid
sequenceDiagram
    participant User
    participant ReasoningTypes
    participant CombinedReasoning
    participant Abduction
    participant Induction
    participant Deduction
    participant ReasoningMemory
    
    User->>ReasoningTypes: create("abduction+induction+deduction")
    ReasoningTypes->>CombinedReasoning: Instantiate with 3 components
    User->>CombinedReasoning: perform_reasoning(input, context)
    
    CombinedReasoning->>Abduction: perform_reasoning(input, context)
    Abduction->>ReasoningMemory: Get context
    ReasoningMemory-->>Abduction: Context tags
    Abduction->>Abduction: Execute abduction
    Abduction->>ReasoningMemory: Store result
    Abduction-->>CombinedReasoning: Return result1
    
    CombinedReasoning->>Induction: perform_reasoning(input, context + result1)
    Induction->>ReasoningMemory: Get updated context
    ReasoningMemory-->>Induction: Context tags
    Induction->>Induction: Execute induction
    Induction->>ReasoningMemory: Store result
    Induction-->>CombinedReasoning: Return result2
    
    CombinedReasoning->>Deduction: perform_reasoning(input, context + result1 + result2)
    Deduction->>ReasoningMemory: Get updated context
    ReasoningMemory-->>Deduction: Context tags
    Deduction->>Deduction: Execute deduction
    Deduction->>ReasoningMemory: Store result
    Deduction-->>CombinedReasoning: Return result3
    
    CombinedReasoning->>CombinedReasoning: _synthesize_result(results)
    CombinedReasoning-->>User: Return combined result
```

## Key Concepts
### 1. Reasoning Type Composition
- Combine up to 3 reasoning types using + syntax
- Types execute in sequence (left to right)
- Output from each type becomes context for the next
- Final result synthesized from all outputs

### 2. Context Passing
Each reasoning step receives:
- Original input arguments
- Initial context (if provided)
- Results from previous steps
- Memory context tags
Context keys:
- prev_<TypeName>_result: Individual step result
- prev_step_result: Last step result
- Memory context tags (e.g., "high_priority_context")

### 3. Memory Integration
- All reasoning types share a common memory
- Memory provides:
    - Contextual tags based on recent experiences
    - Experience prioritization (SumTree)
    - Automatic checkpointing
    - Tag-based experience retrieval

### 4. Result Synthesis
The combined reasoning:
- Collects individual results in combined_result
- Sets reasoning_types to combined type names
- Sets final_output to last result by default
- Can be customized by overriding _synthesize_result()

----

## Usage Examples

### Single Reasoning
```code
reasoning_types = ReasoningTypes()
abduction = reasoning_types.create("abduction")
result = abduction.perform_reasoning(observations="The grass is wet")
```

### Combined Reasoning
```code
combo = reasoning_types.create("abduction+induction")
result = combo.perform_reasoning(
    observations=["Temperature drop", "Wind increase"],
    context={"location": "Seattle"}
)
```

### Custom Synthesis
```code
class CustomCombined(CombinedReasoning):
    def _synthesize_result(self, results):
        return {
            "insights": self._extract_insights(results),
            "confidence": self._calculate_confidence(results)
        }
```
