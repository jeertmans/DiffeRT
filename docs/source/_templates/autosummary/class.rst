{% if module.rsplit('.', 1)[1].startswith('_') -%}
{{ (module.rsplit('.', 1)[0] + '.' + objname) | escape | underline}}
{%- else -%}
{{ fullname | escape | underline}}
{%- endif %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   {% if objname == 'MaterialsDict' -%}
   {#- Behaves just like a plain 'dict' (plus alias resolution), so its
       inherited and overridden dict methods are not worth documenting. -#}
   {%- elif objname == 'InteractionType' -%}
   :members:
   :member-order: bysource
   {%- else -%}
   :members:
   :inherited-members:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods if item != '__init__' %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. rubric:: Detailed documentation
   {%- endif %}
