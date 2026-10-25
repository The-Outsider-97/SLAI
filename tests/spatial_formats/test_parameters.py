"""Expression, unit and dependency regressions for safe parametric SGEO."""
import pytest
from src.utils.spatial_formats.common.errors import ValidationError, ResourceLimitError
from src.utils.spatial_formats.sgeo.parameters import (
    Quantity, evaluate_expression, expression_dependencies, evaluate_parameters,
)
from src.utils.spatial_formats.sgeo.families import generate_geometry
from src.utils.spatial_formats.common.ids import new_id

def test_dimension_aware_math_and_conditional():
    names={"width":Quantity(2.0,(1,0)),"height":Quantity(3.0,(1,0))}
    assert evaluate_expression("width * height",names)==Quantity(6.0,(2,0))
    assert evaluate_expression("width / 2",names)==Quantity(1.0,(1,0))
    assert evaluate_expression("if(width > length(1.5), length(0.12), length(0.08))",names)==Quantity(0.12,(1,0))
    assert expression_dependencies("max(width, height)") == {"width","height"}

def test_dimensional_mismatch_denied():
    with pytest.raises(ValidationError):
        evaluate_expression("width + width * height",{"width":Quantity(2,(1,0)),"height":Quantity(3,(1,0))})
    with pytest.raises(ValidationError):
        evaluate_expression("sin(width)",{"width":Quantity(2,(1,0))})

@pytest.mark.parametrize("expr",[
    "__import__('os').system('echo unsafe')", "a.__class__", "[0] * 10000000",
    "open('file')", "2**100", "1/0", "x[0]", "(lambda: 10)()",
])
def test_malicious_or_invalid_expressions_denied(expr):
    with pytest.raises((ValidationError,ResourceLimitError)):
        evaluate_expression(expr,{"x":Quantity(3)})

def test_dependency_dag_shared_ancestors():
    specs={
      "width":{"type":"length","default":2.0,"min":0.1},
      "height":{"type":"length","default":3.0},
      "half":{"type":"length","expression":"width / 2"},
      "area":{"type":"area","expression":"width * height"},
      "sum":{"type":"length","expression":"half + width"},
      "double":{"type":"length","expression":"sum + half"},
    }
    assert evaluate_parameters(specs)["double"]==4.0
    assert evaluate_parameters(specs,{"width":4.0})["area"]==12.0

def test_cycle_detected():
    specs={"a":{"type":"number","expression":"b+1"},
           "b":{"type":"number","expression":"a+1"}}
    with pytest.raises(ValidationError,match="cyclic"):
        evaluate_parameters(specs)

def test_family_derived_parameter_regeneration():
    family={"id":new_id(),"name":"Derived Block",
            "parameters":{
                "width":{"type":"length","default":2,"min":0.1},
                "depth":{"type":"length","default":1,"min":0.1},
                "height":{"type":"length","expression":"width * 2","min":0.1},
            },
            "generator":{"kind":"builtin.box_v1"}}
    assert generate_geometry(family,{"width":3})["size"]==[3.0,1.0,6.0]
    with pytest.raises(ValidationError):
        generate_geometry(family,{"height":10})

def test_readonly_override_rejected():
    with pytest.raises(ValidationError):
        evaluate_parameters({"x":{"type":"number","default":1,"read_only":True}},{"x":3})

def test_no_arbitrary_functions():
    with pytest.raises(ValidationError):
        evaluate_expression("eval(10)",{})
