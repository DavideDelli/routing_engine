use osmpbf::{Element, ElementReader};
use roaring::RoaringTreemap;
use std::collections::HashMap;
use std::time::Instant;
use rayon::slice::ParallelSliceMut;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::fs::File;      // AGGIUNTO
use std::io::Write;     // AGGIUNTO

// --- STRUTTURE DATI ---

#[derive(Debug, Clone)]
struct Node {
    lat: f32,
    lon: f32,
}

#[derive(Debug, Clone)]
struct Edge {
    target: u32,
    weight: f32,
}

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f32,
    position: u32,
}

impl Eq for State {}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.cost.partial_cmp(&self.cost)
    }
}
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn main() {
    let pbf_path = "blue_banana.osm.pbf";

    // --- PASSATA 1 ---
    println!("Inizio Passata 1: Estrazione ID nodi stradali...");
    let start_1 = Instant::now();
    let reader = ElementReader::from_path(pbf_path).expect("File PBF non trovato");
    
    let nodi_stradali_validi = reader.par_map_reduce(
        |element| {
            let mut local_bitmap = RoaringTreemap::new();
            if let Element::Way(way) = element {
                if is_drivable(&way) {
                    for node_id in way.refs() {
                        local_bitmap.insert(node_id as u64);
                    }
                }
            }
            local_bitmap
        },
        || RoaringTreemap::new(),
        |mut a, b| { a |= b; a },
    ).expect("Errore Passata 1");
    println!("Passata 1 completata in {:.2?}! Nodi da tenere: {}", start_1.elapsed(), nodi_stradali_validi.len());

    // --- PASSATA 2 ---
    println!("Inizio Passata 2: Estrazione coordinate...");
    let start_2 = Instant::now();
    let mut id_mapping: HashMap<u64, u32> = HashMap::with_capacity(nodi_stradali_validi.len() as usize);
    let mut nodes: Vec<Node> = Vec::with_capacity(nodi_stradali_validi.len() as usize);

    let reader = ElementReader::from_path(pbf_path).unwrap();
    let mut internal_id_counter: u32 = 0;

    reader.for_each(|element| {
        if let Element::DenseNode(dense_node) = element {
            let osm_id = dense_node.id() as u64;
            if nodi_stradali_validi.contains(osm_id) {
                nodes.push(Node {
                    lat: dense_node.lat() as f32,
                    lon: dense_node.lon() as f32,
                });
                id_mapping.insert(osm_id, internal_id_counter);
                internal_id_counter += 1;
            }
        }
    }).expect("Errore Passata 2");
    println!("Passata 2 completata in {:.2?}!", start_2.elapsed());

    // --- PASSATA 3 ---
    println!("Inizio Passata 3: Estrazione degli archi stradali...");
    let start_3 = Instant::now();
    let mut raw_edges: Vec<(u32, u32, f32)> = Vec::new();

    let reader = ElementReader::from_path(pbf_path).unwrap();
    reader.for_each(|element| {
        if let Element::Way(way) = element { // SISTEMATO: aggiunto spacchettamento
            if is_drivable(&way) { 
                let (forward, backward) = get_directions(&way);
                let speed_ms = get_speed_kmh(&way) / 3.6;
                
                let refs = way.refs().collect::<Vec<i64>>();
                for window in refs.windows(2) {
                    let source_osm = window[0] as u64;
                    let target_osm = window[1] as u64;

                    if let (Some(&source_id), Some(&target_id)) = (id_mapping.get(&source_osm), id_mapping.get(&target_osm)) {
                        let node1 = &nodes[source_id as usize];
                        let node2 = &nodes[target_id as usize];
                        let distance = haversine_distance(node1.lat, node1.lon, node2.lat, node2.lon);
                        let time_seconds = distance / speed_ms;
                        
                        if forward { raw_edges.push((source_id, target_id, time_seconds)); }
                        if backward { raw_edges.push((target_id, source_id, time_seconds)); } 
                    }
                }
            }
        }
    }).expect("Errore Passata 3");
    println!("Passata 3 completata in {:.2?}!", start_3.elapsed());

    // --- PASSATA 4 (CSR) ---
    println!("Inizio Passata 4: Costruzione del grafo CSR...");
    let start_4 = Instant::now();
    raw_edges.par_sort_unstable_by_key(|edge| edge.0);

    let num_nodes = nodes.len();
    let mut edges: Vec<Edge> = Vec::with_capacity(raw_edges.len());
    let mut offsets: Vec<usize> = vec![0; num_nodes + 1];

    for (source, target, weight) in raw_edges.into_iter() {
        edges.push(Edge { target, weight });
        offsets[source as usize + 1] += 1;
    }
    for i in 0..num_nodes {
        offsets[i + 1] += offsets[i];
    }
    println!("Passata 4 completata in {:.2?}!", start_4.elapsed());

    // --- ANALISI LCC ---
    println!("======================================");
    println!("      ANALISI TOPOLOGIA (LCC)         ");
    println!("======================================");
    let start_lcc = Instant::now();
    let in_lcc = find_lcc(nodes.len(), &edges, &offsets);
    println!("Tempo analisi topologia: {:.2?}", start_lcc.elapsed());

    // --- TEST DI ROUTING ---
    println!("\n======================================");
    println!(" INIZIO TEST DI ROUTING A*");
    println!("======================================");

    let lugano_lat = 46.003; let lugano_lon = 8.951;
    let zurigo_lat = 47.376; let zurigo_lon = 8.541;

    let start_node = find_nearest_valid_node(lugano_lat, lugano_lon, &nodes, &offsets, &in_lcc);
    let goal_node = find_nearest_valid_node(zurigo_lat, zurigo_lon, &nodes, &offsets, &in_lcc);
    
    let start_a_star = Instant::now();
    let result = a_star(start_node, goal_node, &nodes, &edges, &offsets);
    let elapsed_a_star = start_a_star.elapsed();

    if let Some((time_sec, path)) = result {
        println!("✅ PERCORSO TROVATO in {:.2?}", elapsed_a_star);
        println!("Tempo stimato: {:.0}h {:.0}m", time_sec / 3600.0, (time_sec % 3600.0) / 60.0);
        
        let mut geojson = String::from(r#"{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"LineString","coordinates":["#);
        for (i, &node_id) in path.iter().enumerate() {
            let node = &nodes[node_id as usize];
            geojson.push_str(&format!("[{:.6}, {:.6}]", node.lon, node.lat));
            if i < path.len() - 1 { geojson.push(','); }
        }
        geojson.push_str(r#"]},"properties":{"name":"Lugano-Zurigo_HGV"}}]}"#);
        
        let mut file = File::create("route.geojson").unwrap();
        file.write_all(geojson.as_bytes()).unwrap();
        println!("🗺️  File 'route.geojson' generato!");
    } else {
        println!("❌ NESSUN PERCORSO TROVATO");
    }
}

// --- FUNZIONI DI SUPPORTO ---

fn haversine_distance(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let r = 6371000.0;
    let phi1 = lat1.to_radians();
    let phi2 = lat2.to_radians();
    let delta_phi = (lat2 - lat1).to_radians();
    let delta_lambda = (lon2 - lon1).to_radians();
    let a = (delta_phi / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (delta_lambda / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    r * c
}

fn a_star(start: u32, goal: u32, nodes: &[Node], edges: &[Edge], offsets: &[usize]) -> Option<(f32, Vec<u32>)> {
    let mut dist = vec![f32::INFINITY; nodes.len()];
    let mut came_from = vec![u32::MAX; nodes.len()]; 
    let mut heap = BinaryHeap::new();

    dist[start as usize] = 0.0;
    heap.push(State { cost: 0.0, position: start });
    let max_speed_ms = 130.0 / 3.6; 

    while let Some(State { cost: f_cost, position }) = heap.pop() {
        if position == goal {
            let mut path = Vec::new();
            let mut current = goal;
            while current != start {
                path.push(current);
                current = came_from[current as usize];
            }
            path.push(start);
            path.reverse();
            return Some((dist[goal as usize], path));
        }

        if f_cost > dist[position as usize] + (haversine_distance(nodes[position as usize].lat, nodes[position as usize].lon, nodes[goal as usize].lat, nodes[goal as usize].lon) / max_speed_ms) + 0.001 {
            continue;
        }

        let start_edge = offsets[position as usize];
        let end_edge = offsets[position as usize + 1];
        for edge in &edges[start_edge..end_edge] {
            let next_cost = dist[position as usize] + edge.weight;
            if next_cost < dist[edge.target as usize] {
                dist[edge.target as usize] = next_cost;
                came_from[edge.target as usize] = position;
                let h = haversine_distance(nodes[edge.target as usize].lat, nodes[edge.target as usize].lon, nodes[goal as usize].lat, nodes[goal as usize].lon) / max_speed_ms;
                heap.push(State { cost: next_cost + h, position: edge.target });
            }
        }
    }
    None
}

fn get_speed_kmh(way: &osmpbf::Way) -> f32 {
    if let Some((_, speed_str)) = way.tags().find(|(k, _)| *k == "maxspeed") {
        if let Ok(speed) = speed_str.parse::<f32>() { return speed; }
    }
    match way.tags().find(|(k, _)| *k == "highway").map(|(_, v)| v).unwrap_or("") {
        "motorway" => 120.0, "trunk" => 100.0, "primary" => 80.0,
        "secondary" => 60.0, "tertiary" => 50.0, _ => 50.0,
    }
}

fn get_directions(way: &osmpbf::Way) -> (bool, bool) {
    let mut f = true; let mut b = true;
    let is_m = way.tags().any(|(k, v)| k == "highway" && (v == "motorway" || v == "motorway_link"));
    if let Some((_, val)) = way.tags().find(|(k, _)| *k == "oneway") {
        match val { "yes" | "true" | "1" => b = false, "-1" => f = false, _ => {} }
    } else if is_m { b = false; }
    (f, b)
}

fn is_drivable(way: &osmpbf::Way) -> bool {
    way.tags().any(|(k, v)| k == "highway" && matches!(v, "motorway"|"trunk"|"primary"|"secondary"|"tertiary"|"unclassified"|"residential"|"motorway_link"|"trunk_link"|"primary_link"|"secondary_link"|"tertiary_link"))
}

fn find_lcc(nodes_len: usize, edges: &[Edge], offsets: &[usize]) -> Vec<bool> {
    let mut visited = vec![false; nodes_len];
    let mut best_start = 0; let mut max_size = 0;
    for i in 0..nodes_len {
        if !visited[i] {
            let mut stack = vec![i as u32]; visited[i] = true;
            let mut size = 0;
            while let Some(node) = stack.pop() {
                size += 1;
                for edge in &edges[offsets[node as usize]..offsets[node as usize + 1]] {
                    if !visited[edge.target as usize] {
                        visited[edge.target as usize] = true;
                        stack.push(edge.target);
                    }
                }
            }
            if size > max_size { max_size = size; best_start = i as u32; }
        }
    }
    let mut in_lcc = vec![false; nodes_len];
    let mut stack = vec![best_start]; in_lcc[best_start as usize] = true;
    while let Some(node) = stack.pop() {
        for edge in &edges[offsets[node as usize]..offsets[node as usize + 1]] {
            if !in_lcc[edge.target as usize] { in_lcc[edge.target as usize] = true; stack.push(edge.target); }
        }
    }
    println!("LCC: {} nodi", max_size);
    in_lcc
}

fn find_nearest_valid_node(lat: f32, lon: f32, nodes: &[Node], offsets: &[usize], in_lcc: &[bool]) -> u32 {
    let mut best_dist = f32::INFINITY; let mut best_node = 0;
    for (i, node) in nodes.iter().enumerate() {
        if !in_lcc[i] || (offsets[i+1] - offsets[i] == 0) { continue; }
        let d = (node.lat - lat).powi(2) + (node.lon - lon).powi(2);
        if d < best_dist { best_dist = d; best_node = i as u32; }
    }
    best_node
}