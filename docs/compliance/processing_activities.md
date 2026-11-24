# Processing activities (Art. 30)

`ProcessingRegistry` (SQLite, optimistic versioning, history) is the register and the lawful-basis anchor. An activity must name:
purpose, lawful basis **with validated evidence**, data categories, recipients, retention, transfer mechanism (adequacy/SCC/BCR/Art. 49),
and the Art. 9/10 condition where applicable. `export_ropa()` produces the register. Without a registered, active activity the policy engine blocks (GOV-LAW-04).
