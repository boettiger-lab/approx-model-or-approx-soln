""" GENERAL UTILITIES WITHOUT OTHER NATURAL PLACE TO BE """

def dict_pretty_print(D: dict, indent_lvl=0):
  """ RStudio terminal doesn't like json.dumps so do it manual """
  print("Using 3 decimal places.")
  base_indent = indent_lvl * " "
  indent = (indent_lvl+2)*" "
  print(f"{base_indent}" + "{")
  for key, value in D.items():
    print(f"{indent}{key}: ", end="")
    if type(value) is dict:
      print("")
      dict_pretty_print(value, indent_lvl + 2)
    else:
      print(f"{value:.3f}")
  print(f"{base_indent}" + "}")
  
def print_params(env) -> None:
  """ Pretty-prints the parameters  of the env. (env is a gym class with a env.config['parameters'] attribute.) """
  dict_pretty_print(env.config['parameters'])
