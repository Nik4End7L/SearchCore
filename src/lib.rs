use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use quick_xml::events::Event;
use quick_xml::reader::Reader;

struct Product {
    original_name: String,
    price: String,
    oldprice: String,
    id: String,
    url: String,
    picture: String,
    vendor: String,
    created_at: String,
    in_stock: bool,
    category_ids: Vec<String>,
}

#[pyclass]
struct SimpleSearchEngine {
    catalog: HashMap<String, Product>,
    category_map: HashMap<String, String>,
    name_index: HashMap<String, HashSet<String>>,
    desc_index: HashMap<String, HashSet<String>>,
    unique_name_words: HashSet<String>,
    unique_desc_words: HashSet<String>,
    stop_words: HashSet<String>,
}

fn get_close_matches(word: &str, unique_words: &HashSet<String>, n: usize, cutoff: f64) -> Vec<String> {
    let mut matches: Vec<(f64, &String)> = Vec::new();
    for w in unique_words {
        let len_max = word.chars().count().max(w.chars().count()) as f64;
        if len_max == 0.0 { continue; }
        let dist = strsim::levenshtein(word, w) as f64;
        let sim = 1.0 - (dist / len_max);
        if sim >= cutoff {
            matches.push((sim, w));
        }
    }
    matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    matches.into_iter().take(n).map(|(_, w)| w.clone()).collect()
}

fn read_text_event(e: &quick_xml::events::BytesText) -> String {
    e.unescape().map(|c| c.into_owned()).unwrap_or_default()
}

fn read_cdata_event(e: &quick_xml::events::BytesCData) -> String {
    String::from_utf8_lossy(e).into_owned()
}

#[pymethods]
impl SimpleSearchEngine {
    #[new]
    fn new() -> Self {
        let stop_words: HashSet<String> = ["для", "от", "и", "в", "с", "по", "на"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        SimpleSearchEngine {
            catalog: HashMap::new(),
            category_map: HashMap::new(),
            name_index: HashMap::new(),
            desc_index: HashMap::new(),
            unique_name_words: HashSet::new(),
            unique_desc_words: HashSet::new(),
            stop_words,
        }
    }

    fn load_xml(&mut self, file_name: &str) -> PyResult<usize> {
        self.catalog.clear();
        self.category_map.clear();
        self.name_index.clear();
        self.desc_index.clear();
        self.unique_name_words.clear();
        self.unique_desc_words.clear();

        if !Path::new(file_name).exists() { return Ok(0); }

        let mut reader = Reader::from_file(file_name)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        reader.trim_text(true);

        let mut buf = Vec::new();
        let mut in_offer = false;
        let mut current_tag = String::new();

        let mut current_category_id_decl = String::new();
        let mut product_category_ids: Vec<String> = Vec::new();

        let mut name_val = String::new();
        let mut price_val = "—".to_string();
        let mut oldprice_val = String::new();
        let mut id_val = String::new();
        let mut url_val = String::new();
        let mut picture_val = String::new();
        let mut vendor_val = String::new();
        let mut created_at_val = String::new();
        let mut in_stock = true;
        let mut extra_text = String::new();
        let mut fallback_counter = 0;

        macro_rules! handle_text {
            ($text:expr) => {
                if !in_offer && current_tag == "category" && !current_category_id_decl.is_empty() {
                    self.category_map.insert(current_category_id_decl.clone(), $text);
                } else if in_offer && !current_tag.is_empty() {
                    match current_tag.as_str() {
                        "name" | "title" => { name_val = $text; }
                        "price" | "cost" => { price_val = $text; }
                        "oldprice" => { oldprice_val = $text; }
                        "url" => { url_val = $text; }
                        "picture" => { picture_val = $text; }
                        "vendor" | "brand" | "manufacturer" => { vendor_val = $text; }
                        "createdat" | "created_at" | "date" => { created_at_val = $text; }
                        "categoryid" | "category_id" => { product_category_ids.push($text); }
                        "id" | "vendorcode" | "product_id" => {
                            if id_val.is_empty() { id_val = $text; }
                        }
                        "quantity" => {
                            if $text == "0" || $text == "0.0" { in_stock = false; }
                        }
                        "description" | "param" | "category" => {
                            extra_text.push(' ');
                            extra_text.push_str(&$text);
                        }
                        _ => {}
                    }
                }
            };
        }

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().into_inner())
                        .to_lowercase();
                    if !in_offer && tag == "category" {
                        current_category_id_decl = e.attributes()
                            .filter_map(|a| a.ok())
                            .find(|a| a.key.into_inner() == b"id")
                            .map(|a| String::from_utf8_lossy(&a.value).to_string())
                            .unwrap_or_default();
                        current_tag = tag;
                    } else if tag == "offer" || tag == "product" || tag == "item" {
                        in_offer = true;
                        id_val = e.attributes()
                            .filter_map(|a| a.ok())
                            .find(|a| a.key.into_inner() == b"id")
                            .map(|a| String::from_utf8_lossy(&a.value).to_string())
                            .unwrap_or_default();
                        let available_attr = e.attributes()
                            .filter_map(|a| a.ok())
                            .find(|a| a.key.into_inner() == b"available");
                        in_stock = match available_attr {
                            Some(attr) => {
                                let val = String::from_utf8_lossy(&attr.value).to_lowercase();
                                val != "false" && val != "0"
                            }
                            None => true,
                        };
                        name_val.clear();
                        price_val = "—".to_string();
                        oldprice_val.clear();
                        url_val.clear();
                        picture_val.clear();
                        vendor_val.clear();
                        created_at_val.clear();
                        product_category_ids.clear();
                        extra_text.clear();
                    } else if in_offer {
                        current_tag = tag;
                    }
                }

                Ok(Event::Text(ref e)) => {
                    let text = read_text_event(e);
                    handle_text!(text);
                }

                Ok(Event::CData(ref e)) => {
                    let text = read_cdata_event(e);
                    handle_text!(text);
                }

                Ok(Event::End(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().into_inner()).to_lowercase();
                    if !in_offer && tag == "category" {
                        current_category_id_decl.clear();
                        current_tag.clear();
                    } else if tag == "offer" || tag == "product" || tag == "item" {
                        in_offer = false;
                        if !name_val.is_empty() {
                            let low_name = name_val.to_lowercase();
                            let final_id = if id_val.is_empty() {
                                fallback_counter += 1;
                                format!("gen_{}", fallback_counter)
                            } else {
                                id_val.clone()
                            };
                            let safe_id: String = final_id.chars().map(|c| {
                                if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }
                            }).collect();

                            let product = Product {
                                original_name: name_val.clone(),
                                price: price_val.clone(),
                                oldprice: oldprice_val.clone(),
                                id: safe_id,
                                url: url_val.clone(),
                                picture: picture_val.clone(),
                                vendor: vendor_val.clone(),
                                created_at: created_at_val.clone(),
                                in_stock,
                                category_ids: product_category_ids.clone(),
                            };

                            let index_text = if vendor_val.is_empty() {
                                low_name.clone()
                            } else {
                                format!("{} {}", low_name, vendor_val.to_lowercase())
                            };

                            self.catalog.insert(low_name.clone(), product);

                            for word in self.normalize_and_tokenize(&index_text) {
                                self.name_index.entry(word.clone())
                                    .or_insert_with(HashSet::new)
                                    .insert(low_name.clone());
                                self.unique_name_words.insert(word);
                            }
                            if !extra_text.is_empty() {
                                for word in self.normalize_and_tokenize(&extra_text) {
                                    self.desc_index.entry(word.clone())
                                        .or_insert_with(HashSet::new)
                                        .insert(low_name.clone());
                                    self.unique_desc_words.insert(word);
                                }
                            }
                        }
                        current_tag.clear();
                    }
                }

                Ok(Event::Eof) => break,
                Err(e) => return Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
                _ => {}
            }
            buf.clear();
        }
        Ok(self.catalog.len())
    }

    fn search(&self, q_text: String) -> PyResult<Vec<HashMap<String, String>>> {
        let words = self.normalize_and_tokenize(&q_text);
        if words.is_empty() { return Ok(Vec::new()); }

        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut found = false;

        let unique_name_list = &self.unique_name_words;
        let unique_desc_list = &self.unique_desc_words;

        for w in &words {
            let prefix_matches_name: Vec<&String> = unique_name_list.iter()
                .filter(|db_w| db_w.starts_with(w.as_str()))
                .collect();
            let prefix_matches_desc: Vec<&String> = unique_desc_list.iter()
                .filter(|db_w| db_w.starts_with(w.as_str()))
                .collect();

            if !prefix_matches_name.is_empty() {
                found = true;
                for pw in prefix_matches_name {
                    if let Some(pks) = self.name_index.get(pw) {
                        for pk in pks {
                            let score = if pw == w { 100.0 } else { 80.0 };
                            *scores.entry(pk.clone()).or_insert(0.0) += score;
                        }
                    }
                }
            } else {
                let query_starts_with_db: Vec<&String> = unique_name_list.iter()
                    .filter(|db_w| {
                        let db_len = db_w.chars().count();
                        let q_len = w.chars().count();
                        db_len >= 3 && w.starts_with(db_w.as_str()) && db_len as f64 >= q_len as f64 * 0.7
                    })
                    .collect();

                if !query_starts_with_db.is_empty() {
                    found = true;
                    for pw in query_starts_with_db {
                        if let Some(pks) = self.name_index.get(pw) {
                            for pk in pks { *scores.entry(pk.clone()).or_insert(0.0) += 70.0; }
                        }
                    }
                } else {
                    let closest_name = get_close_matches(w, unique_name_list, 3, 0.7);
                    if !closest_name.is_empty() {
                        found = true;
                        for cw in &closest_name {
                            if let Some(pks) = self.name_index.get(cw) {
                                for pk in pks { *scores.entry(pk.clone()).or_insert(0.0) += 60.0; }
                            }
                        }
                    } else {
                        let mut temp_word = w.clone();
                        let mut temp_found = false;
                        while temp_word.chars().count() >= 3 {
                            temp_word.pop();
                            let sub_prefix_name: Vec<&String> = unique_name_list.iter()
                                .filter(|db_w| db_w.starts_with(&temp_word))
                                .collect();
                            if !sub_prefix_name.is_empty() {
                                temp_found = true;
                                found = true;
                                for pw in sub_prefix_name {
                                    if let Some(pks) = self.name_index.get(pw) {
                                        for pk in pks { *scores.entry(pk.clone()).or_insert(0.0) += 40.0; }
                                    }
                                }
                                break;
                            }
                        }
                        if !temp_found && !prefix_matches_desc.is_empty() {
                            found = true;
                            for pw in prefix_matches_desc {
                                if let Some(pks) = self.desc_index.get(pw) {
                                    for pk in pks { *scores.entry(pk.clone()).or_insert(0.0) += 5.0; }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !found || scores.is_empty() { return Ok(Vec::new()); }

        let mut ranked: Vec<(f64, &String)> = Vec::new();
        for (pk, score) in &scores {
            if let Some(prod) = self.catalog.get(pk) {
                let final_score = score + if prod.in_stock { 10.0 } else { 0.0 };
                ranked.push((final_score, pk));
            }
        }
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut results = Vec::new();
        for (_, pk) in ranked.iter().take(500) {
            if let Some(prod) = self.catalog.get(*pk) {
                let mut map = HashMap::new();
                map.insert("id".to_string(), prod.id.clone());
                map.insert("original_name".to_string(), prod.original_name.clone());
                map.insert("price".to_string(), prod.price.clone());
                map.insert("oldprice".to_string(), prod.oldprice.clone());
                map.insert("url".to_string(), prod.url.clone());
                map.insert("picture".to_string(), prod.picture.clone());
                map.insert("vendor".to_string(), prod.vendor.clone());
                map.insert("created_at".to_string(), prod.created_at.clone());
                map.insert("in_stock".to_string(), if prod.in_stock { "true".to_string() } else { "false".to_string() });

                let mut cat_ids = Vec::new();
                let mut cat_names = Vec::new();
                for cid in &prod.category_ids {
                    cat_ids.push(cid.clone());
                    cat_names.push(self.category_map.get(cid).cloned().unwrap_or_default());
                }
                map.insert("category_ids".to_string(), cat_ids.join(","));
                map.insert("category_names".to_string(), cat_names.join("|"));

                results.push(map);
            }
        }

        Ok(results)
    }
}

impl SimpleSearchEngine {
    fn normalize_and_tokenize(&self, text: &str) -> Vec<String> {
        let text = text.to_lowercase();
        let mut normalized = String::new();
        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len() {
            normalized.push(chars[i]);
            if i < chars.len() - 1 {
                let curr = chars[i];
                let next = chars[i + 1];
                if (curr.is_numeric() && next.is_alphabetic()) || (curr.is_alphabetic() && next.is_numeric()) {
                    normalized.push(' ');
                }
            }
        }
        let mut words = Vec::new();
        let mut current_word = String::new();
        for c in normalized.chars() {
            if c.is_alphanumeric() || c == 'ё' {
                current_word.push(c);
            } else if !current_word.is_empty() {
                if current_word.chars().count() >= 2 && !self.stop_words.contains(&current_word) {
                    words.push(current_word.clone());
                }
                current_word.clear();
            }
        }
        if !current_word.is_empty() && current_word.chars().count() >= 2 && !self.stop_words.contains(&current_word) {
            words.push(current_word);
        }
        words
    }
}

#[pymodule]
fn neman_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SimpleSearchEngine>()?;
    Ok(())
}

//Nik4End7L