use osmpbf::{Element, ElementReader};
use roaring::RoaringTreemap;
use std::collections::HashMap;
use std::time::Instant;
use rayon::slice::ParallelSliceMut;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// La nostra struttura dati iper-compatta per il singolo nodo
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

// Struttura che inseriremo nella coda di priorità
#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f32,      // Costo reale accumulato + euristica
    position: u32,  // ID interno del nodo
}

// Implementiamo questi trait per trasformare il BinaryHeap standard (Max-Heap) in un Min-Heap, 
// in modo che estragga sempre il costo più basso.
impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Invertiamo l'ordine di confronto!
        other.cost.partial_cmp(&self.cost)
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn main() {
    // --- PASSATA 1 (Il codice che hai già eseguito) ---
    println!("Inizio Passata 1: Estrazione ID nodi stradali...");
    let start_1 = Instant::now();
    let reader = ElementReader::from_path("switzerland-latest.osm.pbf").expect("File PBF non trovato");
    
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

    // --- PREPARAZIONE PASSATA 2 ---
    println!("Inizio Passata 2: Estrazione coordinate...");
    let start_2 = Instant::now();

    // Pre-allochiamo esattamente la memoria che ci serve per evitare riallocazioni costose
    let capacity = nodi_stradali_validi.len() as usize;
    let mut id_mapping: HashMap<u64, u32> = HashMap::with_capacity(capacity);
    let mut nodes: Vec<Node> = Vec::with_capacity(capacity);

    // Riapriamo il file dall'inizio
    let reader = ElementReader::from_path("switzerland-latest.osm.pbf").unwrap();
    let mut internal_id_counter: u32 = 0;

    // Leggiamo i DenseNode in modo sequenziale per popolare l'array in modo ordinato
    reader.for_each(|element| {
        if let Element::DenseNode(dense_node) = element {
            let osm_id = dense_node.id() as u64;
            
            // Se questo nodo fa parte di una strada (è nel nostro RoaringBitmap)
            if nodi_stradali_validi.contains(osm_id) {
                // 1. Lo salviamo nel vettore principale
                nodes.push(Node {
                    lat: dense_node.lat() as f32, // f32 è sufficiente per la precisione stradale e dimezza la RAM usata
                    lon: dense_node.lon() as f32,
                });
                
                // 2. Creiamo la traduzione OSM_ID -> ID_Interno
                id_mapping.insert(osm_id, internal_id_counter);
                internal_id_counter += 1;
            }
        }
    }).expect("Errore Passata 2");

    println!("Passata 2 completata in {:.2?}!", start_2.elapsed());
    println!("Array Nodi in RAM: {} elementi", nodes.len());
    println!("Dimensione stimata dell'array dei Nodi: {:.2} MB", (nodes.len() * std::mem::size_of::<Node>()) as f64 / 1_048_576.0);

    // --- PASSATA 3: Estrazione degli Archi ---
    println!("Inizio Passata 3: Estrazione degli archi stradali...");
    let start_3 = Instant::now();

    // Creiamo un vettore per ospitare gli archi grezzi: (Sorgente, Destinazione, Costo_in_metri)
    let mut raw_edges: Vec<(u32, u32, f32)> = Vec::new();

    let reader = ElementReader::from_path("switzerland-latest.osm.pbf").unwrap();
    
    reader.for_each(|element| {
        if let Element::Way(way) = element {
            // Controlliamo di nuovo che sia una strada
            if is_drivable(&way) { // CORRETTO QUI: una sola graffa!
                let refs = way.refs().collect::<Vec<i64>>();
                
                // Iteriamo sulle coppie di nodi adiacenti nella strada
                for window in refs.windows(2) {
                    let source_osm = window[0] as u64;
                    let target_osm = window[1] as u64;

                    // Troviamo i nostri ID interni compatti
                    if let (Some(&source_id), Some(&target_id)) = (id_mapping.get(&source_osm), id_mapping.get(&target_osm)) {
                        
                        // Prendiamo le coordinate dei due nodi dal nostro array in RAM
                        let node1 = &nodes[source_id as usize];
                        let node2 = &nodes[target_id as usize];
                        
                        // Calcoliamo la distanza
                        let distance = haversine_distance(node1.lat, node1.lon, node2.lat, node2.lon);
                        
                        // Aggiungiamo l'arco. Nota: OSM definisce i sensi unici (oneway). 
                        // Per ora, per semplicità, aggiungiamo l'arco in entrambe le direzioni (grafo non orientato)
                        raw_edges.push((source_id, target_id, distance));
                        raw_edges.push((target_id, source_id, distance)); 
                    }
                }
            }
        }
    }).expect("Errore Passata 3");

    println!("Passata 3 completata in {:.2?}!", start_3.elapsed());
    println!("Archi totali estratti: {}", raw_edges.len());

    // --- PASSATA 4: Costruzione della Compressed Sparse Row (CSR) ---
    println!("Inizio Passata 4: Costruzione del grafo CSR...");
    let start_4 = Instant::now();

    // 1. Ordiniamo gli archi grezzi per nodo sorgente USANDO TUTTI I CORE!
    raw_edges.par_sort_unstable_by_key(|edge| edge.0);

    let num_nodes = nodes.len();
    let mut edges: Vec<Edge> = Vec::with_capacity(raw_edges.len());
    
    // L'array degli offset ha dimensione N+1 per facilitare la logica del nodo finale
    let mut offsets: Vec<usize> = vec![0; num_nodes + 1];

    // 2. Popoliamo l'array finale degli archi e contiamo le connessioni
    for (source, target, weight) in raw_edges.into_iter() {
        edges.push(Edge { target, weight });
        // Segniamo che il nodo 'source' ha un arco in più.
        // Lo facciamo nell'indice + 1 per preparare la somma cumulativa.
        offsets[source as usize + 1] += 1;
    }

    // 3. Somma cumulativa: trasformiamo i conteggi negli indici di partenza effettivi
    for i in 0..num_nodes {
        offsets[i + 1] += offsets[i];
    }

    println!("Passata 4 completata in {:.2?}!", start_4.elapsed());

    // --- REPORT FINALE MEMORIA ---
    let nodes_mb = (nodes.capacity() * std::mem::size_of::<Node>()) as f64 / 1_048_576.0;
    let edges_mb = (edges.capacity() * std::mem::size_of::<Edge>()) as f64 / 1_048_576.0;
    let offsets_mb = (offsets.capacity() * std::mem::size_of::<usize>()) as f64 / 1_048_576.0;
    
    println!("======================================");
    println!("      MOTORE LOGISTICO PRONTO         ");
    println!("======================================");
    println!("RAM Occupata dai Nodi:    {:.2} MB", nodes_mb);
    println!("RAM Occupata dagli Archi: {:.2} MB", edges_mb);
    println!("RAM Occupata da Offsets:  {:.2} MB", offsets_mb);
    println!("--------------------------------------");
    println!("RAM TOTALE DEL GRAFO:     {:.2} MB", nodes_mb + edges_mb + offsets_mb);

    println!("======================================");
    println!("      ANALISI TOPOLOGIA (LCC)         ");
    println!("======================================");
    let start_lcc = Instant::now();
    let in_lcc = find_lcc(nodes.len(), &edges, &offsets);
    println!("Tempo analisi topologia: {:.2?}", start_lcc.elapsed());

    // --- TEST DI ROUTING: LUGANO -> ZURIGO ---
    println!("\n======================================");
    println!(" INIZIO TEST DI ROUTING A*");
    println!("======================================");

    let lugano_lat = 46.003;
    let lugano_lon = 8.951;
    let zurigo_lat = 47.376;
    let zurigo_lon = 8.541;

    // Aggiungi &in_lcc qui!
    let start_node = find_nearest_valid_node(lugano_lat, lugano_lon, &nodes, &offsets, &in_lcc);
    let goal_node = find_nearest_valid_node(zurigo_lat, zurigo_lon, &nodes, &offsets, &in_lcc);
    println!("Ricerca del percorso in corso...");
    println!("Partenza: Lugano (Nodo ID interno: {})", start_node);
    println!("Arrivo:   Zurigo (Nodo ID interno: {})", goal_node);

    // Facciamo partire il cronometro per l'algoritmo puro
    let start_a_star = Instant::now();
    
    // Invochiamo il nostro motore CSR
    let result = a_star(start_node, goal_node, &nodes, &edges, &offsets);
    
    let elapsed_a_star = start_a_star.elapsed();

    println!("--------------------------------------");
    if let Some(cost) = result {
        println!("✅ PERCORSO TROVATO!");
        println!("Tempo di calcolo netto: {:.2?}", elapsed_a_star);
        // Dividiamo per 1000 perché il nostro costo è in metri
        println!("Distanza totale stimata: {:.2} km", cost / 1000.0); 
    } else {
        println!("❌ NESSUN PERCORSO TROVATO (I nodi potrebbero essere isolati)");
    }
    println!("======================================\n");
}

// Calcola la distanza in metri tra due coordinate usando la formula dell'Haversine
fn haversine_distance(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let r = 6371000.0; // Raggio della Terra in metri
    let phi1 = lat1.to_radians();
    let phi2 = lat2.to_radians();
    let delta_phi = (lat2 - lat1).to_radians();
    let delta_lambda = (lon2 - lon1).to_radians();

    let a = (delta_phi / 2.0).sin().powi(2)
        + phi1.cos() * phi2.cos() * (delta_lambda / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    r * c
}

fn a_star(
    start: u32, 
    goal: u32, 
    nodes: &[Node], 
    edges: &[Edge], 
    offsets: &[usize]
) -> Option<f32> {
    
    // Inizializziamo tutte le distanze all'infinito
    let mut dist = vec![f32::INFINITY; nodes.len()];
    let mut heap = BinaryHeap::new();

    // Partenza
    dist[start as usize] = 0.0;
    heap.push(State { cost: 0.0, position: start });

    while let Some(State { cost: _f_cost, position }) = heap.pop() {
        // Se abbiamo estratto il nodo destinazione, restituiamo il costo REALE g(n)
        if position == goal {
            return Some(dist[goal as usize]);
        }

        // --- BUG RIMOSSO: Nessun controllo f_cost > g_cost qui! ---

        let start_edge = offsets[position as usize];
        let end_edge = offsets[position as usize + 1];

        // Iteriamo in modo contiguo in memoria (cache-hit al 100%)
        for edge in &edges[start_edge..end_edge] {
            let next_cost = dist[position as usize] + edge.weight;

            // Se troviamo un percorso più breve verso il vicino
            if next_cost < dist[edge.target as usize] {
                dist[edge.target as usize] = next_cost;
                
                // Calcoliamo l'euristica h(n)
                let heuristic = haversine_distance(
                    nodes[edge.target as usize].lat, nodes[edge.target as usize].lon,
                    nodes[goal as usize].lat, nodes[goal as usize].lon,
                );
                
                // Inseriamo f(n) = g(n) + h(n) nella coda di priorità
                heap.push(State { 
                    cost: next_cost + heuristic, 
                    position: edge.target 
                });
            }
        }
    }

    None
}

// Controlla se la 'Way' è una strada percorribile da veicoli a motore
fn is_drivable(way: &osmpbf::Way) -> bool {
    way.tags().any(|(key, value)| {
        key == "highway" && matches!(
            value,
            "motorway" | "trunk" | "primary" | "secondary" | "tertiary" | 
            "unclassified" | "residential" | "motorway_link" | "trunk_link" | 
            "primary_link" | "secondary_link" | "tertiary_link"
        )
    })
}

// Trova la Largest Connected Component (LCC) per ignorare le "isole" di dati sporchi
fn find_lcc(nodes_len: usize, edges: &[Edge], offsets: &[usize]) -> Vec<bool> {
    let mut visited = vec![false; nodes_len];
    let mut best_start_node = 0;
    let mut max_component_size = 0;

    // 1. Troviamo il nodo che fa partire la componente più grande
    for i in 0..nodes_len {
        if !visited[i] {
            let mut stack = vec![i as u32];
            visited[i] = true;
            let mut size = 0;

            while let Some(node) = stack.pop() {
                size += 1;
                let start = offsets[node as usize];
                let end = offsets[node as usize + 1];
                for edge in &edges[start..end] {
                    if !visited[edge.target as usize] {
                        visited[edge.target as usize] = true;
                        stack.push(edge.target);
                    }
                }
            }

            if size > max_component_size {
                max_component_size = size;
                best_start_node = i as u32;
            }
        }
    }

    // 2. Facciamo una seconda passata rapida per "colorare" (salvare) solo la LCC
    let mut in_lcc = vec![false; nodes_len];
    let mut stack = vec![best_start_node];
    in_lcc[best_start_node as usize] = true;

    while let Some(node) = stack.pop() {
        let start = offsets[node as usize];
        let end = offsets[node as usize + 1];
        for edge in &edges[start..end] {
            if !in_lcc[edge.target as usize] {
                in_lcc[edge.target as usize] = true;
                stack.push(edge.target);
            }
        }
    }

    println!("Pulizia Grafo: Rete principale identificata! Contiene {} nodi su {}", max_component_size, nodes_len);
    in_lcc
}

// Trova l'ID interno del nodo più vicino che non sia un vicolo cieco
// Modifica la firma e aggiungi il controllo in_lcc
fn find_nearest_valid_node(lat: f32, lon: f32, nodes: &[Node], offsets: &[usize], in_lcc: &[bool]) -> u32 {
    let mut best_dist = f32::INFINITY;
    let mut best_node = 0;

    for (i, node) in nodes.iter().enumerate() {
        // SE IL NODO NON È NELLA RETE PRINCIPALE, IGNORALO!
        if !in_lcc[i] {
            continue;
        }

        let num_edges = offsets[i + 1] - offsets[i];
        if num_edges == 0 { continue; }

        let d_lat = node.lat - lat;
        let d_lon = node.lon - lon;
        let dist_sq = d_lat * d_lat + d_lon * d_lon;

        if dist_sq < best_dist {
            best_dist = dist_sq;
            best_node = i as u32;
        }
    }
    best_node
}