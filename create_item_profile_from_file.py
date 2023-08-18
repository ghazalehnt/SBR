from data_loading_create_profile import create_item_profile_from_file

name = "GR_UCSD"
if name == "GR_UCSD":
	item_file = ""
	output_path = ""
	item_fields = ["title", "genres", "description"]
else:
	item_file = ""
	output_path = ""
	item_fields = ["title", "category", "description"]
case_sensitive = True
normalize_negation = True

create_item_profile_from_file(name, item_file, item_fields, case_sensitive, normalize_negation, output_path)