{% if module.rsplit('.', 1)[1].startswith('_') -%}
{{ (module.rsplit('.', 1)[0] + '.' + objname) | escape | underline}}
{%- else -%}
{{ fullname | escape | underline}}
{%- endif %}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
