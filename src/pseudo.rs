use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormConservingPseudo {
    pub element: String,
    pub functional: String,
    pub valence_charge: f64,
    pub radial_grid: Vec<f64>,
    pub radial_weights: Vec<f64>,
    pub local_potential_ry: Vec<f64>,
    pub projectors: Vec<Projector>,
    pub dij_ry: Vec<Vec<f64>>,
    pub atomic_density: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projector {
    pub index: usize,
    pub angular_momentum: usize,
    pub cutoff_points: usize,
    pub beta: Vec<f64>,
}

fn section<'a>(text: &'a str, name: &str) -> Result<&'a str> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let start = text
        .find(&open)
        .with_context(|| format!("missing {open}"))?
        + open.len();
    let end = text[start..]
        .find(&close)
        .with_context(|| format!("missing {close}"))?
        + start;
    Ok(&text[start..end])
}

fn numbers(text: &str) -> Result<Vec<f64>> {
    text.split_whitespace()
        .map(|token| {
            token
                .replace(['D', 'd'], "E")
                .parse::<f64>()
                .with_context(|| format!("invalid floating-point token {token:?}"))
        })
        .collect()
}

fn repeated_sections<'a>(text: &'a str, name: &str) -> Result<Vec<&'a str>> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let mut remaining = text;
    let mut result = Vec::new();
    while let Some(start) = remaining.find(&open) {
        let body_start = start + open.len();
        let body_end = remaining[body_start..]
            .find(&close)
            .with_context(|| format!("unterminated {open}"))?
            + body_start;
        result.push(&remaining[body_start..body_end]);
        remaining = &remaining[body_end + close.len()..];
    }
    Ok(result)
}

impl NormConservingPseudo {
    /// Read the legacy tagged UPF representation used by QE's Si.pz-vbc.UPF.
    /// XML UPF v2 is rejected until its semantics are independently validated.
    pub fn from_legacy_upf(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read pseudopotential {}", path.display()))?;
        if text.contains("<UPF version=") {
            bail!("UPF v2 XML parsing is not yet validated; legacy Si.pz-vbc.UPF is required");
        }
        let header = section(&text, "PP_HEADER")?;
        let lines: Vec<_> = header
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        ensure!(lines.len() >= 11, "legacy PP_HEADER is truncated");
        let element = lines[1]
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string();
        ensure!(
            lines[2].split_whitespace().next() == Some("NC"),
            "only norm-conserving pseudopotentials are supported"
        );
        ensure!(
            lines[3].split_whitespace().next() == Some("F"),
            "nonlinear core corrections are unsupported"
        );
        let functional = lines[4]
            .split_whitespace()
            .nth(1)
            .unwrap_or_default()
            .to_string();
        ensure!(
            functional == "PZ",
            "only the PZ LDA pseudopotential is supported"
        );
        let valence_charge: f64 = lines[5]
            .split_whitespace()
            .next()
            .context("missing valence charge")?
            .parse()?;
        let mesh_points: usize = lines[9]
            .split_whitespace()
            .next()
            .context("missing mesh size")?
            .parse()?;
        let counts: Vec<usize> = lines[10]
            .split_whitespace()
            .take(2)
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(counts.len() == 2, "missing wavefunction/projector counts");

        let mesh = section(&text, "PP_MESH")?;
        let radial_grid = numbers(section(mesh, "PP_R")?)?;
        let radial_weights = numbers(section(mesh, "PP_RAB")?)?;
        let local_potential_ry = numbers(section(&text, "PP_LOCAL")?)?;
        let atomic_density = numbers(section(&text, "PP_RHOATOM")?)?;
        for (label, values) in [
            ("radial grid", &radial_grid),
            ("radial weights", &radial_weights),
            ("local potential", &local_potential_ry),
            ("atomic density", &atomic_density),
        ] {
            ensure!(
                values.len() == mesh_points,
                "{label} has {} values; expected {mesh_points}",
                values.len()
            );
            ensure!(
                values.iter().all(|x| x.is_finite()),
                "{label} contains non-finite values"
            );
        }
        ensure!(
            radial_grid.windows(2).all(|pair| pair[1] > pair[0]),
            "radial grid is not increasing"
        );

        let nonlocal = section(&text, "PP_NONLOCAL")?;
        let mut projectors = Vec::new();
        for body in repeated_sections(nonlocal, "PP_BETA")? {
            let mut lines = body.lines().map(str::trim).filter(|line| !line.is_empty());
            let metadata: Vec<_> = lines
                .next()
                .context("projector metadata is missing")?
                .split_whitespace()
                .take(2)
                .collect();
            ensure!(metadata.len() == 2, "invalid projector metadata");
            let index = metadata[0].parse()?;
            let angular_momentum = metadata[1].parse()?;
            let cutoff_points: usize = lines
                .next()
                .context("projector cutoff is missing")?
                .split_whitespace()
                .next()
                .context("empty projector cutoff")?
                .parse()?;
            let beta = numbers(&lines.collect::<Vec<_>>().join(" "))?;
            // Legacy UPF stores beta only through `kkbeta`; values beyond the
            // cutoff are exactly zero and are intentionally omitted.
            ensure!(
                beta.len() == cutoff_points,
                "projector {index} has {} points; expected cutoff length {cutoff_points}",
                beta.len()
            );
            ensure!(
                cutoff_points <= mesh_points,
                "projector cutoff exceeds radial mesh"
            );
            projectors.push(Projector {
                index,
                angular_momentum,
                cutoff_points,
                beta,
            });
        }
        ensure!(
            projectors.len() == counts[1],
            "projector count does not match header"
        );

        let mut dij_ry = vec![vec![0.0; projectors.len()]; projectors.len()];
        let dij = section(nonlocal, "PP_DIJ")?;
        let mut lines = dij.lines().map(str::trim).filter(|line| !line.is_empty());
        let nonzero: usize = lines
            .next()
            .context("missing Dij count")?
            .split_whitespace()
            .next()
            .context("empty Dij count")?
            .parse()?;
        let mut seen = 0;
        for line in lines {
            let fields: Vec<_> = line.split_whitespace().collect();
            ensure!(fields.len() >= 3, "invalid Dij row");
            let i: usize = fields[0].parse()?;
            let j: usize = fields[1].parse()?;
            let value: f64 = fields[2].replace(['D', 'd'], "E").parse()?;
            ensure!(
                i > 0 && j > 0 && i <= projectors.len() && j <= projectors.len(),
                "Dij index out of range"
            );
            dij_ry[i - 1][j - 1] = value;
            dij_ry[j - 1][i - 1] = value;
            seen += 1;
        }
        ensure!(seen == nonzero, "Dij nonzero count does not match header");
        Ok(Self {
            element,
            functional,
            valence_charge,
            radial_grid,
            radial_weights,
            local_potential_ry,
            projectors,
            dij_ry,
            atomic_density,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_reference_silicon_pseudo_when_available() {
        let path = Path::new(
            "../dft studies/checkouts/q-e-7.6/EPW/examples/sic/pp/Si.pz-vbc.UPF",
        );
        if !path.exists() {
            return;
        }
        let pseudo = NormConservingPseudo::from_legacy_upf(path).unwrap();
        assert_eq!(pseudo.element, "Si");
        assert_eq!(pseudo.valence_charge, 4.0);
        assert_eq!(pseudo.radial_grid.len(), 431);
        assert_eq!(pseudo.projectors.len(), 2);
        assert_eq!(pseudo.projectors[0].angular_momentum, 0);
        assert_eq!(pseudo.projectors[1].angular_momentum, 1);
    }
}
