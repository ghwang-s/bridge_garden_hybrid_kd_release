from __future__ import annotations

from importlib import import_module

from ..schema import DomainDefinition


def build_domain(name: str) -> DomainDefinition:
    module = import_module(f"bridge_garden_v2.domains.{name}")
    return module.build_domain()
