"""Targeted mobile layout checks — real bugs only, no header touch-target noise."""

from __future__ import annotations

from typing import Any

TARGETED_MOBILE_CHECKS = """
() => {
  const findings = [];
  const path = window.location.pathname;
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  if (document.documentElement.scrollWidth > vw + 8) {
    findings.push({
      id: 'horizontal_overflow',
      severity: 'major',
      title: 'Horizontal overflow on ' + path,
      description:
        'Document scrollWidth (' +
        document.documentElement.scrollWidth +
        'px) exceeds viewport (' +
        vw +
        'px).',
    });
  }

  const fixedBottom = document.querySelector(
    'footer.fixed, nav.fixed, [class*="fixed"][class*="bottom-0"]',
  );
  if (fixedBottom && /\\/learn\\/modules\\//.test(path)) {
    const fr = fixedBottom.getBoundingClientRect();
    if (fr.height > 24 && fr.bottom >= vh - 4) {
      const obscured = Array.from(
        document.querySelectorAll(
          'button, [role="radio"], input[type="radio"], label, [data-testid*="quiz"]',
        ),
      ).filter((el) => {
        const r = el.getBoundingClientRect();
        if (r.width < 12 || r.height < 12) return false;
        return r.bottom > fr.top + 6 && r.top < vh - 2;
      });
      if (obscured.length >= 2) {
        findings.push({
          id: 'bottom_chrome_overlap',
          severity: 'major',
          title: 'Fixed bottom chrome obscures interactive content',
          description:
            obscured.length +
            ' interactive elements sit under or behind the fixed bottom bar on the module reader.',
        });
      }
    }
  }

  if (path.includes('/settings')) {
    const bottomNav = document.querySelector('nav');
    if (bottomNav) {
      const nr = bottomNav.getBoundingClientRect();
      if (nr.bottom >= vh - 4 && nr.height > 40) {
        const underNav = Array.from(document.querySelectorAll('button')).filter((el) => {
          const r = el.getBoundingClientRect();
          return r.height >= 28 && r.bottom > nr.top + 2 && r.top < nr.bottom;
        });
        if (underNav.length >= 2) {
          findings.push({
            id: 'settings_under_bottom_nav',
            severity: 'minor',
            title: 'Settings controls may sit under bottom navigation',
            description:
              underNav.length +
              ' setting controls overlap the fixed bottom nav region; scroll may not reveal them fully.',
          });
        }
      }
    }
  }

  if (path === '/studio/materials') {
    const rows = Array.from(document.querySelectorAll('a[href*="/studio/materials/"]')).slice(
      0,
      6,
    );
    for (const row of rows) {
      const title = row.querySelector('h2, h3, [class*="title"], p');
      const actions = row.querySelectorAll('button');
      if (!title || actions.length === 0) continue;
      const tr = title.getBoundingClientRect();
      for (const btn of actions) {
        const br = btn.getBoundingClientRect();
        if (br.width < 8 || br.height < 8) continue;
        const overlaps =
          br.left < tr.right &&
          br.right > tr.left &&
          br.top < tr.bottom &&
          br.bottom > tr.top;
        if (overlaps) {
          findings.push({
            id: 'materials_action_overlap',
            severity: 'major',
            title: 'Material list action buttons overlap card content',
            description:
              'Generate/delete actions overlap the material title on /studio/materials at 390px.',
          });
          break;
        }
        if (br.right > vw - 2) {
          findings.push({
            id: 'materials_action_clipped',
            severity: 'major',
            title: 'Material list action buttons clip off-screen',
            description: 'Action buttons extend past the right viewport edge on /studio/materials.',
          });
          break;
        }
      }
      if (findings.some((f) => f.id.startsWith('materials_'))) break;
    }
  }

  return { viewport: vw, findings };
}
"""


def findings_to_issues(findings: list[dict[str, Any]], url: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in findings:
        issue_id = str(item.get("id") or "")
        if issue_id in seen:
            continue
        seen.add(issue_id)
        issues.append(
            {
                "severity": item.get("severity") or "minor",
                "title": item.get("title") or f"Mobile layout issue on {url}",
                "description": item.get("description") or "",
                "repro_steps": [
                    f"Open {url} on a 390px mobile viewport",
                    "Scroll the page and inspect layout overlap / overflow",
                ],
            }
        )
    return issues
