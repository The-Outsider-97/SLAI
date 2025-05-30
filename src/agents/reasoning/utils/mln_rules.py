import datetime
from collections import defaultdict

# --- Helper Function (assuming it would be part of your ValidationEngine or accessible) ---
def fact_exists(kb, s, p, o, min_confidence=0.7):
    """Checks if a fact exists in the KB with at least min_confidence."""
    return kb.get((s, p, o), 0.0) >= min_confidence

def get_fact_value(kb, s, p, min_confidence=0.0):
    """Gets the object of a fact if it exists with min_confidence, else None."""
    for (subj, pred, obj), conf in kb.items():
        if subj == s and pred == p and conf >= min_confidence:
            return obj
    return None

# --- MLN-style Soft Rules ---
mln_rules = [
    # --- Basic Contradictions ---
    {
        "id": "R001",
        "description": "A being cannot be both alive and dead simultaneously.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "is_alive", "True", conf) and fact_exists(kb, x, "is_dead", "True", conf)
            for x, _, _ in kb if fact_exists(kb, x, "is_alive", "True", conf)  # Iterate over things that are alive
        ),
        "example_violation": {("Socrates", "is_alive", "True"): 0.9, ("Socrates", "is_dead", "True"): 0.95}
    },
    {
        "id": "R002",
        "description": "An object cannot be both a liquid and a solid at the same time and conditions.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "state_is", "Liquid", conf) and fact_exists(kb, x, "state_is", "Solid", conf)
            for x, _, _ in kb if fact_exists(kb, x, "state_is", "Liquid", conf)
        ),
        "example_violation": {("Water_Sample1", "state_is", "Liquid"): 0.8, ("Water_Sample1", "state_is", "Solid"): 0.85}
    },
    {
        "id": "R003",
        "description": "A statement cannot be asserted as both True and False with high confidence.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, s, p, "True", conf) and fact_exists(kb, s, p, "False", conf)
            for s, p, o_val in kb if o_val == "True"  # Iterate facts asserted as True
        ),
        "example_violation": {("Sky", "is_color", "Blue", "True"): 0.9, ("Sky", "is_color", "Blue", "False"): 0.9}
    },
    {
        "id": "R004",
        "description": "An entity cannot be located in two distinct, mutually exclusive places simultaneously.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "located_in", place1, conf) and \
            fact_exists(kb, x, "located_in", place2, conf) and \
            place1 != place2 and \
            not fact_exists(kb, place1, "is_part_of", place2, 0.1) and \
            not fact_exists(kb, place2, "is_part_of", place1, 0.1) # Ensure places aren't hierarchical
            for x, _, _ in kb if fact_exists(kb, x, "located_in", get_fact_value(kb, x, "located_in") or "", conf)
            for _, _, place1 in [(s,pr,o) for s,pr,o in kb if s==x and pr=="located_in"]
            for _, _, place2 in [(s,pr,o) for s,pr,o in kb if s==x and pr=="located_in" and o != place1]
        ),
        "example_violation": {("EiffelTower", "located_in", "Paris"): 1.0, ("EiffelTower", "located_in", "London"): 0.9}
    },

    # --- Relationships & Properties ---
    {
        "id": "R005",
        "description": "If A is a parent of B, B cannot be a parent of A (anti-symmetric parental relationship).",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, a, "is_parent_of", b, conf) and fact_exists(kb, b, "is_parent_of", a, conf)
            for a, p, b in kb if p == "is_parent_of"
        ),
        "example_violation": {("John", "is_parent_of", "Mary"): 0.9, ("Mary", "is_parent_of", "John"): 0.8}
    },
    {
        "id": "R006",
        "description": "If A is married to B, B should also be married to A (symmetric marriage). Violation if not.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, a, "is_married_to", b, conf) and not fact_exists(kb, b, "is_married_to", a, conf * 0.8) # Allow slightly lower conf for symmetry
            for a, p, b in kb if p == "is_married_to"
        ),
        "example_violation": {("Alice", "is_married_to", "Bob"): 0.95, ("Bob", "is_married_to", "Carol"): 0.9} # Bob married to two people
                                                                          # or {("Alice", "is_married_to", "Bob"): 0.95} but no ("Bob", "is_married_to", "Alice")
    },
    {
        "id": "R007",
        "description": "A person cannot be married to more than one person at a time (in most common legal contexts).",
        "lambda_rule": lambda kb, conf: any(
            len([(s,p,o) for s,p,o in kb if s == person and p == "is_married_to" and fact_exists(kb,s,p,o,conf)]) > 1
            for person, _, _ in kb if fact_exists(kb, person, "is_married_to", get_fact_value(kb,person,"is_married_to") or "", conf)
        ),
        "example_violation": {("Bob", "is_married_to", "Alice"): 0.9, ("Bob", "is_married_to", "Carol"): 0.85}
    },
    {
        "id": "R008",
        "description": "If X is_a Mammal, and Mammal is_a Animal, X should not be stated as not_an_Animal.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "is_a", "Mammal", conf) and \
            fact_exists(kb, "Mammal", "is_a", "Animal", conf) and \
            fact_exists(kb, x, "is_a", "not_Animal", conf) # or (x, "is_not_type", "Animal")
            for x, _, _ in kb if fact_exists(kb, x, "is_a", "Mammal", conf)
        ),
        "example_violation": {("Dog", "is_a", "Mammal"): 1.0, ("Mammal", "is_a", "Animal"): 1.0, ("Dog", "is_a", "not_Animal"): 0.9}
    },
    {
        "id": "R009",
        "description": "If X is part_of Y, Y cannot be part_of X (anti-symmetric part-whole).",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "is_part_of", y, conf) and fact_exists(kb, y, "is_part_of", x, conf)
            for x, p, y in kb if p == "is_part_of"
        ),
        "example_violation": {("Engine", "is_part_of", "Car"): 1.0, ("Car", "is_part_of", "Engine"): 0.8}
    },
    {
        "id": "R010",
        "description": "If X is_heavier_than Y, Y cannot be_heavier_than X.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "is_heavier_than", y, conf) and fact_exists(kb, y, "is_heavier_than", x, conf)
            for x, p, y in kb if p == "is_heavier_than"
        ),
        "example_violation": {("Elephant", "is_heavier_than", "Mouse"): 1.0, ("Mouse", "is_heavier_than", "Elephant"): 0.7}
    },

    # --- Temporal Consistency ---
    {
        "id": "R011",
        "description": "A person's birth date cannot be after their death date.",
        "lambda_rule": lambda kb, conf: any(
            (bd_obj := get_fact_value(kb, person, "has_birth_date", conf)) and \
            (dd_obj := get_fact_value(kb, person, "has_death_date", conf)) and \
            (lambda b,d: datetime.datetime.strptime(b, '%Y-%m-%d') > datetime.datetime.strptime(d, '%Y-%m-%d') if isinstance(b,str) and isinstance(d,str) else False)(bd_obj, dd_obj)
            for person, _, _ in kb if get_fact_value(kb, person, "has_birth_date") # Iterate persons with birth_date
        ),
        "example_violation": {("OldMan", "has_birth_date", "1900-01-01"): 1.0, ("OldMan", "has_death_date", "1890-12-31"): 0.9}
    },
    {
        "id": "R012",
        "description": "An event cannot end before it starts.",
        "lambda_rule": lambda kb, conf: any(
            (st_obj := get_fact_value(kb, event, "starts_at_time", conf)) and \
            (et_obj := get_fact_value(kb, event, "ends_at_time", conf)) and \
            (lambda s,e: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")) > datetime.datetime.fromisoformat(e.replace("Z", "+00:00")) if isinstance(s,str) and isinstance(e,str) else False)(st_obj, et_obj)
            for event, _, _ in kb if get_fact_value(kb, event, "starts_at_time")
        ),
        "example_violation": {("MeetingX", "starts_at_time", "2023-01-01T10:00:00Z"): 0.9, ("MeetingX", "ends_at_time", "2023-01-01T09:00:00Z"): 0.9}
    },
    {
        "id": "R013",
        "description": "If EventA occurs_before EventB, EventB cannot occur_before EventA.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, event_a, "occurs_before", event_b, conf) and \
            fact_exists(kb, event_b, "occurs_before", event_a, conf)
            for event_a, p, event_b in kb if p == "occurs_before"
        ),
        "example_violation": {("WW1", "occurs_before", "WW2"): 1.0, ("WW2", "occurs_before", "WW1"): 0.7}
    },

    # --- Common Sense & Domain Knowledge ---
    {
        "id": "R014",
        "description": "Humans are mortal (should not be 'is_immortal' True).",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "is_a", "Human", conf) and fact_exists(kb, x, "is_immortal", "True", conf)
            for x, _, _ in kb if fact_exists(kb, x, "is_a", "Human", conf)
        ),
        "example_violation": {("Bob", "is_a", "Human"): 0.9, ("Bob", "is_immortal", "True"): 0.8}
    },
    {
        "id": "R015",
        "description": "Birds typically can_fly (violation if Bird and 'can_fly' is False, unless specified otherwise e.g. Penguin).",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "is_a", "Bird", conf) and \
            fact_exists(kb, x, "can_fly", "False", conf) and \
            not fact_exists(kb, x, "is_a", "Penguin", 0.5) and \
            not fact_exists(kb, x, "is_a", "Ostrich", 0.5) # Add other flightless birds
            for x, _, _ in kb if fact_exists(kb, x, "is_a", "Bird", conf)
        ),
        "example_violation": {("Sparrow", "is_a", "Bird"): 1.0, ("Sparrow", "can_fly", "False"): 0.9}
    },
    {
        "id": "R016",
        "description": "Water is a liquid (at standard room temperature).",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "is_substance", "Water", conf) and \
            fact_exists(kb, x, "has_temperature_celsius", temp_str, conf) and \
            (lambda t: 5 < int(t) < 40)(temp_str) and # Room temp range
            fact_exists(kb, x, "state_is", "Solid", conf) # Contradiction: solid water at room temp
            for x, _, _ in kb if fact_exists(kb, x, "is_substance", "Water", conf)
            for _, _, temp_str in [(s,pr,o) for s,pr,o in kb if s==x and pr=="has_temperature_celsius"]
        ),
        "example_violation": {("H2O_sample", "is_substance", "Water"): 1.0, ("H2O_sample", "has_temperature_celsius", "25"): 1.0, ("H2O_sample", "state_is", "Solid"): 0.8}
    },
    {
        "id": "R017",
        "description": "An object cannot be transparent and opaque simultaneously.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "has_property", "Transparent", conf) and fact_exists(kb, x, "has_property", "Opaque", conf)
            for x, _, _ in kb if fact_exists(kb, x, "has_property", "Transparent", conf)
        ),
        "example_violation": {("GlassPane", "has_property", "Transparent"): 0.9, ("GlassPane", "has_property", "Opaque"): 0.8}
    },
     {
        "id": "R018",
        "description": "If A causes B, and B causes C, then A should not directly cause not_C.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, a, "causes", b, conf) and \
            fact_exists(kb, b, "causes", c, conf) and \
            fact_exists(kb, a, "causes", f"not_{c}", conf)
            for a, p1, b in kb if p1 == "causes"
            for _, p2, c in kb if p2 == "causes" and _ == b
        ),
        "example_violation": {("Rain", "causes", "WetGround"): 0.9, ("WetGround", "causes", "MuddyShoes"):0.8, ("Rain", "causes", "not_MuddyShoes"): 0.7}
    },
    {
        "id": "R019",
        "description": "A physical object cannot be in a container that it contains.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, item, "is_inside", container, conf) and \
            fact_exists(kb, container, "is_inside", item, conf)
            for item, p, container in kb if p == "is_inside"
        ),
        "example_violation": {("BoxA", "is_inside", "BoxB"): 0.9, ("BoxB", "is_inside", "BoxA"): 0.9}
    },
    {
        "id": "R020",
        "description": "An entity that is singular cannot also be plural.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "has_number", "Singular", conf) and fact_exists(kb, x, "has_number", "Plural", conf)
            for x, _, _ in kb if fact_exists(kb, x, "has_number", "Singular", conf)
        ),
        "example_violation": {("Cat", "has_number", "Singular"): 0.9, ("Cat", "has_number", "Plural"): 0.8}
    },
    {
        "id": "R021",
        "description": "If X is_male, X cannot be_female.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "has_gender", "Male", conf) and fact_exists(kb, x, "has_gender", "Female", conf)
            for x, _, _ in kb if fact_exists(kb, x, "has_gender", "Male", conf)
        ),
        "example_violation": {("Alex", "has_gender", "Male"): 0.9, ("Alex", "has_gender", "Female"): 0.85}
    },
    {
        "id": "R022",
        "description": "A city is typically located_in a country, not the other way around (unless it's a city-state).",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, city, "is_a", "City", conf) and \
            fact_exists(kb, country, "is_a", "Country", conf) and \
            fact_exists(kb, country, "located_in", city, conf) and \
            not fact_exists(kb, city, "is_a", "CityState", 0.5) # Exception for city-states
            for city, p1, _ in kb if p1 == "is_a" and _ == "City"
            for country, p2, _ in kb if p2 == "is_a" and _ == "Country"
        ),
        "example_violation": {("Paris", "is_a", "City"):1.0, ("France", "is_a", "Country"):1.0, ("France", "located_in", "Paris"):0.7}
    },
    {
        "id": "R023",
        "description": "If something has a color, that color should not be 'Transparent' if it's also 'Opaque'.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "has_color", "Transparent", conf) and \
            fact_exists(kb, x, "has_property", "Opaque", conf)
            for x, _, _ in kb if fact_exists(kb, x, "has_color", "Transparent", conf)
        ),
        "example_violation": {("Brick", "has_color", "Transparent"): 0.8, ("Brick", "has_property", "Opaque"): 1.0}
    },
    {
        "id": "R024",
        "description": "An employee works_for a company; a company does not typically work_for an employee.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, employee, "is_a", "Person", conf) and \
            fact_exists(kb, company, "is_a", "Company", conf) and \
            fact_exists(kb, company, "works_for", employee, conf)
            for employee, p1, _ in kb if p1=="is_a" and _=="Person"
            for company, p2, _ in kb if p2=="is_a" and _=="Company"
        ),
        "example_violation": {("JohnDoe", "is_a", "Person"):1.0, ("AcmeCorp", "is_a", "Company"):1.0, ("AcmeCorp", "works_for", "JohnDoe"):0.7}
    },
    {
        "id": "R025",
        "description": "If X is_greater_than Y, and Y is_greater_than Z, then X should not be_less_than Z (transitivity violation).",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "is_greater_than", y, conf) and \
            fact_exists(kb, y, "is_greater_than", z, conf) and \
            fact_exists(kb, x, "is_less_than", z, conf)
            for x, p1, y in kb if p1 == "is_greater_than"
            for _, p2, z in kb if p2 == "is_greater_than" and _ == y
        ),
        "example_violation": {("A", "is_greater_than", "B"):0.9, ("B", "is_greater_than", "C"):0.9, ("A", "is_less_than", "C"):0.8}
    },
    {
        "id": "R026",
        "description": "An action cannot be both possible and impossible.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, action, "is_status", "Possible", conf) and \
            fact_exists(kb, action, "is_status", "Impossible", conf)
            for action, _, _ in kb if fact_exists(kb, action, "is_status", "Possible", conf)
        ),
        "example_violation": {("FlyToMarsTomorrow", "is_status", "Possible"):0.7, ("FlyToMarsTomorrow", "is_status", "Impossible"):0.9}
    },
    {
        "id": "R027",
        "description": "A bachelor is an unmarried man. Contradiction if stated as married or not male.",
        "lambda_rule": lambda kb, conf: any(
            (fact_exists(kb, x, "is_a", "Bachelor", conf) and fact_exists(kb, x, "is_married_to", get_fact_value(kb,x,"is_married_to") or "_any_", conf)) or \
            (fact_exists(kb, x, "is_a", "Bachelor", conf) and fact_exists(kb, x, "has_gender", "Female", conf))
            for x, _, _ in kb if fact_exists(kb, x, "is_a", "Bachelor", conf)
        ),
        "example_violation": {("John", "is_a", "Bachelor"):0.9, ("John", "is_married_to", "Jane"):0.8}
    },
    {
        "id": "R028",
        "description": "If X requires Y, and Y is absent, then X cannot be achieved/true.",
         "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, x, "requires", y, conf) and \
            fact_exists(kb, y, "is_present", "False", conf) and \
            fact_exists(kb, x, "is_achieved", "True", conf)
            for x, p, y in kb if p == "requires"
        ),
        "example_violation": {("Car", "requires", "Fuel"):1.0, ("Fuel", "is_present", "False"):0.9, ("Car", "is_achieved", "True"):0.8} # "is_achieved" here means "is_running"
    },
    {
        "id": "R029",
        "description": "Numbers cannot be both even and odd.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, num, "has_property", "Even", conf) and \
            fact_exists(kb, num, "has_property", "Odd", conf)
            for num, _, _ in kb if fact_exists(kb, num, "has_property", "Even", conf)
        ),
        "example_violation": {("Number7", "has_property", "Even"):0.8, ("Number7", "has_property", "Odd"):0.9}
    },
    {
        "id": "R030",
        "description": "A vegetarian does not eat meat.",
        "lambda_rule": lambda kb, conf: any(
            fact_exists(kb, person, "is_a", "Vegetarian", conf) and \
            fact_exists(kb, person, "eats", "Meat", conf)
            for person, _, _ in kb if fact_exists(kb, person, "is_a", "Vegetarian", conf)
        ),
        "example_violation": {("Alice", "is_a", "Vegetarian"):1.0, ("Alice", "eats", "Meat"):0.7}
    },
    # Add more rules following these patterns...
]

# --- Main execution for testing (optional) ---
if __name__ == "__main__":
    # Sample Knowledge Base (KB)
    # Keys are (subject, predicate, object) tuples, values are confidence scores
    sample_kb = {
        ("Socrates", "is_a", "Human"): 0.99,
        ("Socrates", "is_alive", "True"): 0.05, # Low confidence, might be historical
        ("Socrates", "is_dead", "True"): 0.98,
        # ("Socrates", "is_immortal", "True"): 0.7, # Potential violation with R014
        ("Bob", "is_a", "Human"): 0.9,
        ("Bob", "is_immortal", "True"): 0.8, # R014 Violation
        ("Alice", "is_married_to", "Bob"): 0.95,
        # ("Bob", "is_married_to", "Alice"): 0.92, # Symmetric - OK
        ("Bob", "is_married_to", "Carol"): 0.85, # R007 Violation
        ("Cat", "is_a", "Mammal"): 1.0,
        ("Mammal", "is_a", "Animal"): 1.0,
        ("Cat", "is_a", "not_Animal"): 0.9, # R008 Violation
        ("Engine", "is_part_of", "Car"): 1.0,
        ("Car", "is_part_of", "Engine"): 0.8, # R009 Violation
        ("OldMan", "has_birth_date", "1900-01-01"): 1.0,
        ("OldMan", "has_death_date", "1890-12-31"): 0.9, # R011 Violation
        ("Water_Sample1", "state_is", "Liquid"): 0.8, 
        ("Water_Sample1", "state_is", "Solid"): 0.85 # R002 Violation
    }

    print("Validating Sample Knowledge Base with MLN-style Rules:")
    print("=" * 50)
    confidence_threshold = 0.7 # Only consider facts with this confidence or higher for rule triggers

    violations_found = defaultdict(list)

    for rule_def in mln_rules:
        rule_id = rule_def["id"]
        description = rule_def["description"]
        lambda_func = rule_def["lambda_rule"]
        example_violation_kb = rule_def.get("example_violation", {})

        # Test with sample_kb
        if lambda_func(sample_kb, confidence_threshold):
            violations_found[rule_id].append(f"Rule VIOLATED with sample_kb: {description}")

        # Test with example_violation_kb (if provided)
        if example_violation_kb:
            if lambda_func(example_violation_kb, confidence_threshold):
                 violations_found[rule_id].append(f"Rule CORRECTLY IDENTIFIED example violation: {description}")
            else:
                 violations_found[rule_id].append(f"Rule FAILED to identify example violation: {description} - CHECK LAMBDA OR EXAMPLE")


    if violations_found:
        for rule_id, msgs in violations_found.items():
            for msg in msgs:
                print(f"[{rule_id}] {msg}")
    else:
        print("No violations found in the sample KB based on the defined rules and threshold.")

    print("\nNote: Some rules might require specific date/time parsing or more complex logic than simple string checks.")
