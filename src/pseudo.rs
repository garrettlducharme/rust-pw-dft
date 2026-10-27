use anyhow::{Context, Result, ensure};
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

fn opening_tag<'a>(text: &'a str, name: &str) -> Result<&'a str> {
    let marker = format!("<{name}");
    let start = text
        .find(&marker)
        .with_context(|| format!("missing <{name}>"))?;
    let end = text[start..]
        .find('>')
        .with_context(|| format!("unterminated <{name}>"))?
        + start
        + 1;
    Ok(&text[start..end])
}

fn attribute<'a>(tag: &'a str, name: &str) -> Result<&'a str> {
    let marker = format!("{name}=");
    let start = tag
        .find(&marker)
        .with_context(|| format!("missing attribute {name}"))?
        + marker.len();
    let remaining = tag[start..].trim_start();
    let quote = remaining
        .chars()
        .next()
        .filter(|character| *character == '\'' || *character == '"')
        .with_context(|| format!("attribute {name} is not quoted"))?;
    let value = &remaining[quote.len_utf8()..];
    let end = value
        .find(quote)
        .with_context(|| format!("attribute {name} is unterminated"))?;
    Ok(value[..end].trim())
}

fn tagged_body<'a>(text: &'a str, name: &str) -> Result<&'a str> {
    let marker = format!("<{name}");
    let start = text
        .find(&marker)
        .with_context(|| format!("missing <{name}>"))?;
    let body_start = text[start..]
        .find('>')
        .with_context(|| format!("unterminated <{name}>"))?
        + start
        + 1;
    let close = format!("</{name}>");
    let body_end = text[body_start..]
        .find(&close)
        .with_context(|| format!("missing {close}"))?
        + body_start;
    Ok(&text[body_start..body_end])
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
    /// Read either the pinned legacy silicon representation or a strict
    /// scalar/non-relativistic norm-conserving UPF v2 PZ or PBE representation.
    pub fn from_upf(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read pseudopotential {}", path.display()))?;
        if text.contains("<UPF version=") {
            return Self::from_v2_text(&text);
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

    fn from_v2_text(text: &str) -> Result<Self> {
        let root = opening_tag(text, "UPF")?;
        ensure!(
            attribute(root, "version")?.starts_with("2."),
            "only UPF v2 is supported"
        );
        let header = opening_tag(text, "PP_HEADER")?;
        ensure!(
            attribute(header, "pseudo_type")? == "NC",
            "only norm-conserving UPF v2 pseudopotentials are supported"
        );
        ensure!(
            attribute(header, "is_ultrasoft")?.eq_ignore_ascii_case("F")
                && attribute(header, "is_paw")?.eq_ignore_ascii_case("F"),
            "ultrasoft and PAW UPF v2 pseudopotentials are unsupported"
        );
        ensure!(
            attribute(header, "core_correction")?.eq_ignore_ascii_case("F"),
            "nonlinear core corrections are unsupported"
        );
        ensure!(
            attribute(header, "has_so")?.eq_ignore_ascii_case("F"),
            "spin-orbit UPF v2 pseudopotentials are unsupported"
        );
        let relativistic = attribute(header, "relativistic")?;
        ensure!(
            relativistic.eq_ignore_ascii_case("scalar")
                || relativistic.eq_ignore_ascii_case("no")
                || relativistic.eq_ignore_ascii_case("nonrelativistic"),
            "fully relativistic UPF v2 pseudopotentials are unsupported"
        );
        let functional = attribute(header, "functional")?.to_string();
        ensure!(
            functional.eq_ignore_ascii_case("PZ") || functional.eq_ignore_ascii_case("PBE"),
            "only PZ LDA and PBE pseudopotentials are supported"
        );
        let element = attribute(header, "element")?.to_string();
        ensure!(!element.is_empty(), "UPF v2 element is empty");
        let valence_charge: f64 = attribute(header, "z_valence")?.parse()?;
        let mesh_points: usize = attribute(header, "mesh_size")?.parse()?;
        let projector_count: usize = attribute(header, "number_of_proj")?.parse()?;
        ensure!(mesh_points > 1, "UPF v2 radial mesh is too short");
        ensure!(projector_count > 0, "UPF v2 contains no projectors");

        let radial_grid = numbers(tagged_body(text, "PP_R")?)?;
        let radial_weights = numbers(tagged_body(text, "PP_RAB")?)?;
        let local_potential_ry = numbers(tagged_body(text, "PP_LOCAL")?)?;
        let atomic_density = numbers(tagged_body(text, "PP_RHOATOM")?)?;
        for (label, values) in [
            ("radial grid", &radial_grid),
            ("radial weights", &radial_weights),
            ("local potential", &local_potential_ry),
            ("atomic density", &atomic_density),
        ] {
            ensure!(
                values.len() == mesh_points,
                "UPF v2 {label} has {} values; expected {mesh_points}",
                values.len()
            );
            ensure!(
                values.iter().all(|value| value.is_finite()),
                "UPF v2 {label} contains non-finite values"
            );
        }
        ensure!(
            radial_grid.windows(2).all(|pair| pair[1] > pair[0]),
            "UPF v2 radial grid is not increasing"
        );

        let mut projectors = Vec::with_capacity(projector_count);
        for projector_index in 1..=projector_count {
            let name = format!("PP_BETA.{projector_index}");
            let tag = opening_tag(text, &name)?;
            let parsed_index: usize = attribute(tag, "index")?.parse()?;
            ensure!(
                parsed_index == projector_index,
                "UPF v2 projector indices are not contiguous"
            );
            let angular_momentum: usize = attribute(tag, "angular_momentum")?.parse()?;
            let cutoff_points: usize = attribute(tag, "cutoff_radius_index")?.parse()?;
            ensure!(
                cutoff_points > 0 && cutoff_points <= mesh_points,
                "UPF v2 projector cutoff exceeds the radial mesh"
            );
            let beta = numbers(tagged_body(text, &name)?)?;
            ensure!(
                beta.len() == mesh_points,
                "UPF v2 projector {projector_index} has {} values; expected {mesh_points}",
                beta.len()
            );
            ensure!(
                beta.iter().all(|value| value.is_finite()),
                "UPF v2 projector {projector_index} contains non-finite values"
            );
            projectors.push(Projector {
                index: parsed_index,
                angular_momentum,
                cutoff_points,
                beta,
            });
        }
        let flat_dij = numbers(tagged_body(text, "PP_DIJ")?)?;
        ensure!(
            flat_dij.len() == projector_count * projector_count,
            "UPF v2 Dij has the wrong size"
        );
        ensure!(
            flat_dij.iter().all(|value| value.is_finite()),
            "UPF v2 Dij contains non-finite values"
        );
        for row in 0..projector_count {
            for column in 0..projector_count {
                ensure!(
                    (flat_dij[row * projector_count + column]
                        - flat_dij[column * projector_count + row])
                        .abs()
                        <= 1.0e-10,
                    "UPF v2 Dij is not symmetric"
                );
            }
        }
        let dij_ry = flat_dij
            .chunks_exact(projector_count)
            .map(<[f64]>::to_vec)
            .collect();
        ensure!(
            valence_charge.is_finite() && valence_charge > 0.0,
            "UPF v2 valence charge is invalid"
        );
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
        let path = Path::new("pseudopotentials/Si.pz-vbc.UPF");
        if !path.exists() {
            return;
        }
        let pseudo = NormConservingPseudo::from_upf(path).unwrap();
        assert_eq!(pseudo.element, "Si");
        assert_eq!(pseudo.valence_charge, 4.0);
        assert_eq!(pseudo.radial_grid.len(), 431);
        assert_eq!(pseudo.projectors.len(), 2);
        assert_eq!(pseudo.projectors[0].angular_momentum, 0);
        assert_eq!(pseudo.projectors[1].angular_momentum, 1);
    }

    #[test]
    fn parses_strict_norm_conserving_upf_v2() {
        let text = r#"
<UPF version="2.0.1">
<PP_HEADER element="Si" pseudo_type="NC" relativistic="scalar"
 is_ultrasoft="F" is_paw="F" core_correction="F" has_so="F"
 functional="PZ" z_valence="4.0" mesh_size="3" number_of_proj="1"/>
<PP_MESH>
<PP_R>0.0 0.5 1.0</PP_R>
<PP_RAB>0.25 0.5 0.5</PP_RAB>
</PP_MESH>
<PP_LOCAL>-2.0 -1.0 -0.5</PP_LOCAL>
<PP_NONLOCAL>
<PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="2">
0.0 1.0 0.0
</PP_BETA.1>
<PP_DIJ>1.5</PP_DIJ>
</PP_NONLOCAL>
<PP_RHOATOM>0.0 2.0 0.0</PP_RHOATOM>
</UPF>
"#;
        let pseudo = NormConservingPseudo::from_v2_text(text).unwrap();
        assert_eq!(pseudo.element, "Si");
        assert_eq!(pseudo.functional, "PZ");
        assert_eq!(pseudo.valence_charge, 4.0);
        assert_eq!(pseudo.radial_grid, [0.0, 0.5, 1.0]);
        assert_eq!(pseudo.projectors.len(), 1);
        assert_eq!(pseudo.projectors[0].cutoff_points, 2);
        assert_eq!(pseudo.dij_ry, [vec![1.5]]);
    }

    #[test]
    fn parses_strict_pbe_norm_conserving_upf_v2() {
        let text = r#"
<UPF version="2.0.1">
<PP_HEADER element="Si" pseudo_type="NC" relativistic="scalar"
 is_ultrasoft="F" is_paw="F" core_correction="F" has_so="F"
 functional="PBE" z_valence="4.0" mesh_size="3" number_of_proj="1"/>
<PP_MESH>
<PP_R>0.0 0.5 1.0</PP_R>
<PP_RAB>0.25 0.5 0.5</PP_RAB>
</PP_MESH>
<PP_LOCAL>-2.0 -1.0 -0.5</PP_LOCAL>
<PP_NONLOCAL>
<PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="2">
0.0 1.0 0.0
</PP_BETA.1>
<PP_DIJ>1.5</PP_DIJ>
</PP_NONLOCAL>
<PP_RHOATOM>0.0 2.0 0.0</PP_RHOATOM>
</UPF>
"#;
        let pseudo = NormConservingPseudo::from_v2_text(text).unwrap();
        assert_eq!(pseudo.element, "Si");
        assert_eq!(pseudo.functional, "PBE");
        assert_eq!(pseudo.valence_charge, 4.0);
        assert_eq!(pseudo.projectors.len(), 1);
    }

    #[test]
    fn parses_pinned_upf_v2_and_rejects_out_of_scope_v2_when_available() {
        let lda_path =
            Path::new("../dft studies/checkouts/q-e-7.6/EPW/examples/gan/pp/N_ONCV_LDA-1.0.upf");
        if lda_path.exists() {
            let pseudo = NormConservingPseudo::from_upf(lda_path).unwrap();
            assert_eq!(pseudo.element, "N");
            assert_eq!(pseudo.functional, "PZ");
            assert_eq!(pseudo.valence_charge, 5.0);
            assert_eq!(pseudo.radial_grid.len(), 1058);
            assert_eq!(pseudo.projectors.len(), 4);
            assert!((pseudo.dij_ry[0][0] - 13.970_499_110).abs() < 1.0e-9);
        }

        let spin_orbit_path = Path::new("../dft studies/checkouts/q-e-7.6/pseudo/Si_r.upf");
        if spin_orbit_path.exists() {
            let error = NormConservingPseudo::from_upf(spin_orbit_path).unwrap_err();
            assert!(error.to_string().contains("core corrections"));
        }

        let pbe_path = Path::new(
            "../dft studies/checkouts/q-e-7.6/external/wannier90/pseudo/Si_ONCV_PBE-1.0.upf",
        );
        if pbe_path.exists() {
            let pseudo = NormConservingPseudo::from_upf(pbe_path).unwrap();
            assert_eq!(pseudo.element, "Si");
            assert_eq!(pseudo.functional, "PBE");
            assert_eq!(pseudo.valence_charge, 4.0);
            assert_eq!(pseudo.radial_grid.len(), 602);
            assert_eq!(pseudo.projectors.len(), 4);
        }
    }
}
