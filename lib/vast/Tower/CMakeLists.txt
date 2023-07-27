add_library( vast_tower STATIC
  Tower.cpp
)

add_library(vast::tower ALIAS vast_tower)
