#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <random>
#include <cmath>
#include <memory>
#include <variant>
#include <optional>
#include <regex>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <queue>
#include <stack>

namespace ansi {
    const std::string reset   = "\033[0m";
    const std::string bold    = "\033[1m";
    const std::string dim     = "\033[2m";
    const std::string red     = "\033[31m";
    const std::string green   = "\033[32m";
    const std::string yellow  = "\033[33m";
    const std::string blue    = "\033[34m";
    const std::string magenta = "\033[35m";
    const std::string cyan    = "\033[36m";
    const std::string white   = "\033[37m";
    const std::string gray    = "\033[90m";
}

static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

double randf(double lo = -1.0, double hi = 1.0) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double sigmoid_d(double x) { double s = sigmoid(x); return s * (1.0 - s); }
double relu(double x) { return x > 0 ? x : 0.0; }
double relu_d(double x) { return x > 0 ? 1.0 : 0.0; }
double tanh_act(double x) { return std::tanh(x); }
double tanh_d(double x) { double t = std::tanh(x); return 1.0 - t * t; }

enum class Activation { Sigmoid, ReLU, Tanh, Linear };

struct Layer {
    int in_size, out_size;
    Activation act;
    std::vector<std::vector<double>> W;
    std::vector<double> b, z, a, delta;

    Layer(int in, int out, Activation act) : in_size(in), out_size(out), act(act),
        W(out, std::vector<double>(in)), b(out, 0.0), z(out), a(out), delta(out) {
        double scale = std::sqrt(2.0 / in);
        for (auto& row : W)
            for (auto& w : row)
                w = randf(-scale, scale);
    }

    std::vector<double> forward(const std::vector<double>& x) {
        for (int i = 0; i < out_size; i++) {
            z[i] = b[i];
            for (int j = 0; j < in_size; j++) z[i] += W[i][j] * x[j];
            switch (act) {
                case Activation::Sigmoid: a[i] = sigmoid(z[i]); break;
                case Activation::ReLU:    a[i] = relu(z[i]); break;
                case Activation::Tanh:    a[i] = tanh_act(z[i]); break;
                case Activation::Linear:  a[i] = z[i]; break;
            }
        }
        return a;
    }

    double activate_d(int i) {
        switch (act) {
            case Activation::Sigmoid: return sigmoid_d(z[i]);
            case Activation::ReLU:    return relu_d(z[i]);
            case Activation::Tanh:    return tanh_d(z[i]);
            case Activation::Linear:  return 1.0;
        }
        return 1.0;
    }
};

struct NeuralNet {
    std::vector<Layer> layers;
    double lr;
    int epoch_count = 0;

    NeuralNet() : lr(0.01) {}

    void add_layer(int in, int out, Activation act = Activation::ReLU) {
        layers.emplace_back(in, out, act);
    }

    std::vector<double> forward(const std::vector<double>& x) {
        std::vector<double> cur = x;
        for (auto& l : layers) cur = l.forward(cur);
        return cur;
    }

    double train(const std::vector<double>& x, const std::vector<double>& y) {
        forward(x);
        int n = layers.size();

        auto& last = layers[n - 1];
        double loss = 0.0;
        for (int i = 0; i < last.out_size; i++) {
            double err = last.a[i] - y[i];
            loss += err * err;
            last.delta[i] = err * last.activate_d(i);
        }

        for (int li = n - 2; li >= 0; li--) {
            auto& cur = layers[li];
            auto& nxt = layers[li + 1];
            for (int i = 0; i < cur.out_size; i++) {
                double sum = 0.0;
                for (int j = 0; j < nxt.out_size; j++)
                    sum += nxt.W[j][i] * nxt.delta[j];
                cur.delta[i] = sum * cur.activate_d(i);
            }
        }

        const std::vector<double>* prev = &x;
        for (int li = 0; li < n; li++) {
            auto& l = layers[li];
            for (int i = 0; i < l.out_size; i++) {
                for (int j = 0; j < l.in_size; j++)
                    l.W[i][j] -= lr * l.delta[i] * (*prev)[j];
                l.b[i] -= lr * l.delta[i];
            }
            prev = &l.a;
        }

        epoch_count++;
        return loss / last.out_size;
    }

    std::string info() const {
        std::ostringstream os;
        os << "Layers: " << layers.size() << " | LR: " << lr << " | Epochs trained: " << epoch_count;
        return os.str();
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        int n = layers.size();
        f.write((char*)&n, sizeof(n));
        for (const auto& l : layers) {
            f.write((char*)&l.in_size, sizeof(int));
            f.write((char*)&l.out_size, sizeof(int));
            int a = (int)l.act;
            f.write((char*)&a, sizeof(int));
            for (const auto& row : l.W)
                f.write((char*)row.data(), row.size() * sizeof(double));
            f.write((char*)l.b.data(), l.b.size() * sizeof(double));
        }
        f.write((char*)&lr, sizeof(double));
        f.write((char*)&epoch_count, sizeof(int));
    }

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        int n;
        f.read((char*)&n, sizeof(n));
        layers.clear();
        for (int i = 0; i < n; i++) {
            int in, out, a;
            f.read((char*)&in, sizeof(int));
            f.read((char*)&out, sizeof(int));
            f.read((char*)&a, sizeof(int));
            layers.emplace_back(in, out, (Activation)a);
            auto& l = layers.back();
            for (auto& row : l.W)
                f.read((char*)row.data(), row.size() * sizeof(double));
            f.read((char*)l.b.data(), l.b.size() * sizeof(double));
        }
        f.read((char*)&lr, sizeof(double));
        f.read((char*)&epoch_count, sizeof(int));
        return true;
    }
};

struct Term {
    std::string functor;
    std::vector<std::string> args;

    std::string to_string() const {
        if (args.empty()) return functor;
        std::string s = functor + "(";
        for (size_t i = 0; i < args.size(); i++) {
            if (i) s += ", ";
            s += args[i];
        }
        return s + ")";
    }

    bool operator==(const Term& o) const {
        return functor == o.functor && args == o.args;
    }

    bool is_var(const std::string& s) const {
        return !s.empty() && s[0] == '?';
    }
};

struct Fact {
    Term term;
    std::string source;
    std::chrono::system_clock::time_point added_at;
    double confidence;

    Fact() : confidence(1.0), added_at(std::chrono::system_clock::now()) {}
    Fact(Term t, std::string src, double conf = 1.0)
        : term(std::move(t)), source(std::move(src)),
          added_at(std::chrono::system_clock::now()), confidence(conf) {}
};

struct Condition {
    Term pattern;
};

struct Rule {
    std::string name;
    std::vector<Condition> conditions;
    std::vector<std::string> conclusion_parts;
    int fire_count = 0;
};

struct Derivation {
    std::string conclusion;
    std::string rule_name;
    std::vector<std::string> premises;
    std::chrono::system_clock::time_point derived_at;
};

using Binding = std::map<std::string, std::string>;

bool is_var(const std::string& s) { return !s.empty() && s[0] == '?'; }

std::optional<Binding> unify(const Term& pattern, const Term& fact, Binding bindings = {}) {
    if (is_var(pattern.functor)) {
        auto it = bindings.find(pattern.functor);
        if (it != bindings.end()) { if (it->second != fact.functor) return std::nullopt; }
        else bindings[pattern.functor] = fact.functor;
    } else if (pattern.functor != fact.functor) return std::nullopt;
    if (pattern.args.size() != fact.args.size()) return std::nullopt;
    for (size_t i = 0; i < pattern.args.size(); i++) {
        const auto& pa = pattern.args[i];
        const auto& fa = fact.args[i];
        if (is_var(pa)) {
            auto it = bindings.find(pa);
            if (it != bindings.end()) {
                if (it->second != fa) return std::nullopt;
            } else {
                bindings[pa] = fa;
            }
        } else if (pa != fa) {
            return std::nullopt;
        }
    }
    return bindings;
}

std::string apply_binding(const std::string& s, const Binding& b) {
    if (is_var(s)) {
        auto it = b.find(s);
        return it != b.end() ? it->second : s;
    }
    return s;
}

struct KnowledgeBase {
    std::vector<Fact> facts;
    std::vector<Rule> rules;
    std::vector<Derivation> derivations;
    std::vector<std::string> history;
    std::map<std::string, std::vector<std::string>> concept_graph;

    std::optional<int> find_fact(const Term& t) {
        for (int i = 0; i < (int)facts.size(); i++) {
            if (facts[i].term.functor == t.functor && facts[i].term.args == t.args)
                return i;
        }
        return std::nullopt;
    }

    std::string add_fact(const Term& t, const std::string& source = "user", double conf = 1.0) {
        auto idx = find_fact(t);
        if (idx) {
            std::string old = facts[*idx].term.to_string();
            facts[*idx] = Fact(t, source, conf);
            history.push_back("Overwrote: " + old + " -> " + t.to_string());
            return "OVERWRITE:" + old;
        }
        facts.emplace_back(t, source, conf);
        history.push_back("Added fact: " + t.to_string());
        concept_graph[t.functor];
        if (!t.args.empty()) {
            concept_graph[t.functor].push_back(t.args[0]);
            concept_graph[t.args[0]];
        }
        return "NEW";
    }

    bool add_rule(const Rule& r) {
        for (const auto& existing : rules)
            if (existing.name == r.name) return false;
        rules.push_back(r);
        history.push_back("Added rule: " + r.name);
        return true;
    }

    bool fact_exists(const std::string& desc) {
        for (const auto& d : derivations)
            if (d.conclusion == desc) return true;
        return false;
    }

    std::vector<Binding> match_condition(const Condition& cond, const Binding& current) {
        std::vector<Binding> results;
        Term resolved = cond.pattern;
        resolved.functor = apply_binding(resolved.functor, current);
        for (auto& arg : resolved.args)
            arg = apply_binding(arg, current);

        for (const auto& fact : facts) {
            auto res = unify(resolved, fact.term, current);
            if (res) results.push_back(*res);
        }
        return results;
    }

    std::vector<Binding> solve_conditions(const std::vector<Condition>& conditions) {
        std::vector<Binding> solutions = {{}};
        for (const auto& cond : conditions) {
            std::vector<Binding> next;
            for (const auto& binding : solutions) {
                auto matches = match_condition(cond, binding);
                for (auto& m : matches) next.push_back(m);
            }
            solutions = next;
            if (solutions.empty()) break;
        }
        return solutions;
    }

    std::vector<Derivation> think() {
        std::vector<Derivation> new_derivations;
        bool changed = true;

        while (changed) {
            changed = false;
            for (auto& rule : rules) {
                auto solutions = solve_conditions(rule.conditions);

                for (const auto& sol : solutions) {
                    std::string conclusion;
                    for (const auto& part : rule.conclusion_parts)
                        conclusion += apply_binding(part, sol) + " ";
                    if (!conclusion.empty()) conclusion.pop_back();

                    bool self_ref = false;
                    if (rule.conditions.size() >= 2) {
                        std::set<std::string> first_vals, second_vals;
                        for (const auto& [k, v] : sol) {
                            (first_vals.empty() ? first_vals : second_vals).insert(v);
                        }
                    }

                    if (!fact_exists(conclusion)) {
                        Derivation d;
                        d.conclusion = conclusion;
                        d.rule_name = rule.name;
                        d.derived_at = std::chrono::system_clock::now();
                        for (const auto& [k, v] : sol)
                            d.premises.push_back(k + "=" + v);
                        derivations.push_back(d);
                        new_derivations.push_back(d);
                        history.push_back("Derived: " + conclusion + " via " + rule.name);
                        changed = true;
                        rule.fire_count++;
                    }
                }
            }
        }

        return new_derivations;
    }

    bool query(const std::string& q) {
        for (const auto& d : derivations)
            if (d.conclusion == q) return true;
        for (const auto& f : facts)
            if (f.term.to_string() == q) return true;
        return false;
    }

    std::vector<std::string> explain(const std::string& q) {
        std::vector<std::string> chain;
        for (const auto& d : derivations) {
            if (d.conclusion == q) {
                chain.push_back("Rule \"" + d.rule_name + "\" derived: " + d.conclusion);
                for (const auto& p : d.premises)
                    chain.push_back("  Binding: " + p);

                for (const auto& rule : rules) {
                    if (rule.name == d.rule_name) {
                        for (const auto& cond : rule.conditions) {
                            for (const auto& f : facts) {
                                auto res = unify(cond.pattern, f.term);
                                if (res) chain.push_back("  Fact used: " + f.term.to_string() +
                                    " (source: " + f.source + ")");
                            }
                        }
                    }
                }
                return chain;
            }
        }

        for (const auto& f : facts) {
            if (f.term.to_string() == q) {
                chain.push_back("Direct fact: " + f.term.to_string());
                chain.push_back("  Source: " + f.source);
                return chain;
            }
        }

        chain.push_back("Not found in knowledge base.");
        return chain;
    }

    std::vector<std::string> infer_analogies() {
        std::vector<std::string> suggestions;
        std::map<std::string, std::vector<std::pair<std::string,std::string>>> prop_map;

        for (const auto& f : facts) {
            if (f.term.args.size() == 2)
                prop_map[f.term.args[0]].push_back({f.term.functor, f.term.args[1]});
        }

        auto subjects = std::vector<std::string>();
        for (const auto& [k, _] : prop_map) subjects.push_back(k);

        for (size_t i = 0; i < subjects.size(); i++) {
            for (size_t j = i + 1; j < subjects.size(); j++) {
                const auto& a = prop_map[subjects[i]];
                const auto& b = prop_map[subjects[j]];
                int shared = 0;
                for (const auto& pa : a)
                    for (const auto& pb : b)
                        if (pa == pb) shared++;
                if (shared > 0)
                    suggestions.push_back(subjects[i] + " ~ " + subjects[j] +
                        " (share " + std::to_string(shared) + " properties)");
            }
        }
        return suggestions;
    }

    std::map<std::string, int> concept_frequency() const {
        std::map<std::string, int> freq;
        for (const auto& f : facts) {
            freq[f.term.functor]++;
            for (const auto& a : f.term.args) freq[a]++;
        }
        return freq;
    }

    void forget_weak(double threshold = 0.3) {
        facts.erase(std::remove_if(facts.begin(), facts.end(),
            [threshold](const Fact& f){ return f.confidence < threshold; }),
            facts.end());
    }

    void decay_confidence(double rate = 0.01) {
        for (auto& f : facts)
            if (f.source != "user") f.confidence -= rate;
    }
};

Term parse_term(const std::string& s) {
    Term t;
    auto lp = s.find('(');
    if (lp == std::string::npos) {
        t.functor = s;
        return t;
    }
    t.functor = s.substr(0, lp);
    std::string args_str = s.substr(lp + 1, s.rfind(')') - lp - 1);
    std::istringstream ss(args_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        while (!tok.empty() && tok.front() == ' ') tok = tok.substr(1);
        while (!tok.empty() && tok.back() == ' ') tok.pop_back();
        if (!tok.empty()) t.args.push_back(tok);
    }
    return t;
}

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::vector<std::string> split_tokens(const std::string& s) {
    std::vector<std::string> tokens;
    std::istringstream ss(s);
    std::string tok;
    while (ss >> tok) tokens.push_back(tok);
    return tokens;
}

struct SimpleEmbedding {
    std::unordered_map<std::string, std::vector<float>> table;
    int dim;

    SimpleEmbedding(int d = 16) : dim(d) {}

    std::vector<float>& get_or_create(const std::string& word) {
        auto it = table.find(word);
        if (it == table.end()) {
            auto& v = table[word];
            v.resize(dim);
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            for (auto& x : v) x = dist(rng);
            return v;
        }
        return it->second;
    }

    float similarity(const std::string& a, const std::string& b) {
        auto& va = get_or_create(a);
        auto& vb = get_or_create(b);
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < dim; i++) { dot += va[i]*vb[i]; na += va[i]*va[i]; nb += vb[i]*vb[i]; }
        if (na == 0 || nb == 0) return 0.0f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    void update(const std::string& a, const std::string& b, float signal, float lr = 0.05f) {
        auto& va = get_or_create(a);
        auto& vb = get_or_create(b);
        for (int i = 0; i < dim; i++) {
            va[i] += lr * signal * vb[i];
            vb[i] += lr * signal * va[i];
        }
    }
};

struct MemoryStore {
    struct Entry {
        std::string key, value;
        double weight;
        std::chrono::system_clock::time_point ts;
    };

    std::vector<Entry> entries;
    int max_size;

    MemoryStore(int ms = 512) : max_size(ms) {}

    void store(const std::string& key, const std::string& value, double weight = 1.0) {
        for (auto& e : entries) {
            if (e.key == key) { e.value = value; e.weight = weight; e.ts = std::chrono::system_clock::now(); return; }
        }
        if ((int)entries.size() >= max_size) {
            auto min_it = std::min_element(entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b){ return a.weight < b.weight; });
            *min_it = {key, value, weight, std::chrono::system_clock::now()};
        } else {
            entries.push_back({key, value, weight, std::chrono::system_clock::now()});
        }
    }

    std::optional<std::string> recall(const std::string& key) {
        for (const auto& e : entries)
            if (e.key == key) return e.value;
        return std::nullopt;
    }

    std::vector<Entry> top_k(int k) {
        auto sorted = entries;
        std::sort(sorted.begin(), sorted.end(), [](const Entry& a, const Entry& b){ return a.weight > b.weight; });
        if ((int)sorted.size() > k) sorted.resize(k);
        return sorted;
    }
};

std::string current_time_str() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%H:%M:%S");
    return ss.str();
}

Rule parse_rule(const std::string& line) {
    Rule r;
    std::regex name_re(R"(\"([^\"]+)\")");
    std::smatch m;
    if (std::regex_search(line, m, name_re)) r.name = m[1];
    else r.name = "rule_" + std::to_string(rand());

    size_t if_pos = line.find(" if ");
    size_t then_pos = line.find(" then ");
    if (if_pos == std::string::npos || then_pos == std::string::npos) return r;

    std::string cond_str = trim(line.substr(if_pos + 4, then_pos - if_pos - 4));
    std::string conc_str = trim(line.substr(then_pos + 6));

    std::regex and_re(R"(\s+and\s+)");
    std::sregex_token_iterator cit(cond_str.begin(), cond_str.end(), and_re, -1), cend;
    for (; cit != cend; ++cit) {
        std::string cs = trim(*cit);
        if (!cs.empty()) {
            Condition c;
            c.pattern = parse_term(cs);
            r.conditions.push_back(c);
        }
    }

    std::istringstream ss(conc_str);
    std::string tok;
    while (ss >> tok) r.conclusion_parts.push_back(tok);

    return r;
}

struct CrappyBat {
    NeuralNet nn;
    KnowledgeBase kb;
    SimpleEmbedding emb;
    MemoryStore mem;
    std::string session_name;
    bool verbose_nn = false;
    int cmd_count = 0;

    CrappyBat() : emb(32), mem(256) {
        nn.lr = 0.01;
        nn.add_layer(8, 32, Activation::ReLU);
        nn.add_layer(32, 16, Activation::Tanh);
        nn.add_layer(16, 4, Activation::Sigmoid);
        session_name = "Pipsqueak";
    }

    void print_banner() {
        std::cout << ansi::cyan << ansi::bold;
        std::cout << R"(
  ██████╗██████╗  █████╗ ██████╗ ██████╗ ██╗   ██╗██████╗  █████╗ ████████╗
 ██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗╚══██╔══╝
 ██║     ██████╔╝███████║██████╔╝██████╔╝ ╚████╔╝ ██████╔╝███████║   ██║   
 ██║     ██╔══██╗██╔══██║██╔═══╝ ██╔═══╝   ╚██╔╝  ██╔══██╗██╔══██║   ██║   
 ╚██████╗██║  ██║██║  ██║██║     ██║        ██║   ██████╔╝██║  ██║   ██║   
  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝        ╚═╝   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   
)" << ansi::reset;
        std::cout << ansi::gray << "  Neural Reasoning Engine v2.0 | type 'help' for commands\n\n" << ansi::reset;
    }

    void print_prompt() {
        std::cout << ansi::bold << ansi::blue << ">>>" << ansi::reset << " ";
        std::cout.flush();
    }

    void cmd_add_fact(const std::string& args) {
        std::string trimmed = trim(args);
        Term t = parse_term(trimmed);
        if (t.functor.empty()) {
            std::cout << ansi::red << "✗ Invalid fact syntax.\n" << ansi::reset;
            return;
        }
        std::string result = kb.add_fact(t, "user");
        if (result.starts_with("OVERWRITE:")) {
            std::cout << ansi::yellow << "⚠ Overwriting fact: " << result.substr(10)
                      << " → " << t.to_string() << "\n" << ansi::reset;
        } else {
            std::cout << ansi::green << "✓ Added: " << t.to_string() << "\n" << ansi::reset;
        }
        emb.update(t.functor, t.args.empty() ? "" : t.args[0], 1.0f);
        mem.store(t.to_string(), "fact", 1.0);

        std::vector<double> feat(8, 0.0);
        for (size_t i = 0; i < t.functor.size() && i < 4; i++)
            feat[i] = (double)t.functor[i] / 128.0;
        for (size_t i = 0; i < t.args.size() && i < 2; i++)
            for (size_t j = 0; j < t.args[i].size() && j < 2; j++)
                feat[4 + i*2 + j] = (double)t.args[i][j] / 128.0;
        std::vector<double> target = {1.0, 0.0, 0.0, 0.0};
        nn.train(feat, target);
    }

    void cmd_add_rule(const std::string& args) {
        Rule r = parse_rule(args);
        if (r.name.empty() || r.conditions.empty() || r.conclusion_parts.empty()) {
            std::cout << ansi::red << "✗ Rule parse failed. Format: add_rule \"name\" if cond1 and cond2 then conclusion\n" << ansi::reset;
            return;
        }
        if (kb.add_rule(r)) {
            std::cout << ansi::green << "✓ Added rule: " << r.name << "\n" << ansi::reset;
            std::cout << ansi::gray << "  Conditions: " << r.conditions.size()
                      << " | Conclusion parts: " << r.conclusion_parts.size() << "\n" << ansi::reset;
        } else {
            std::cout << ansi::yellow << "⚠ Rule \"" << r.name << "\" already exists.\n" << ansi::reset;
        }
    }

    void cmd_think() {
        std::cout << ansi::magenta << "Reasoning...\n" << ansi::reset;
        auto derived = kb.think();
        if (derived.empty()) {
            std::cout << ansi::gray << "  - No new conclusions.\n" << ansi::reset;
        } else {
            for (const auto& d : derived) {
                std::cout << ansi::green << "  ✦ Derived via [" << d.rule_name << "]: "
                          << d.conclusion << "\n" << ansi::reset;
            }
        }

        auto analogies = kb.infer_analogies();
        if (!analogies.empty()) {
            std::cout << ansi::cyan << "\n  Analogies detected:\n" << ansi::reset;
            for (const auto& a : analogies)
                std::cout << ansi::cyan << "  ~ " << a << "\n" << ansi::reset;
        }

        kb.decay_confidence(0.005);
    }

    void cmd_query(const std::string& args) {
        std::string q = trim(args);
        bool result = kb.query(q);
        if (result) {
            std::cout << ansi::green << "✓ TRUE\n" << ansi::reset;
            auto chain = kb.explain(q);
            for (const auto& step : chain)
                std::cout << ansi::gray << "  " << step << "\n" << ansi::reset;
        } else {
            for (const auto& f : kb.facts) {
                if (f.term.to_string() == q) {
                    std::cout << ansi::green << "✓ TRUE (direct fact)\n" << ansi::reset;
                    std::cout << ansi::gray << "  Source: " << f.source
                              << " | Confidence: " << std::fixed << std::setprecision(2) << f.confidence << "\n" << ansi::reset;
                    return;
                }
            }
            std::cout << ansi::red << "✗ FALSE (not found)\n" << ansi::reset;
        }
    }

    void cmd_explain(const std::string& args) {
        std::string q = trim(args);
        auto chain = kb.explain(q);
        std::cout << ansi::cyan << "Chain of reasoning:\n" << ansi::reset;
        for (size_t i = 0; i < chain.size(); i++)
            std::cout << ansi::white << "  " << (i+1) << ". " << chain[i] << "\n" << ansi::reset;
    }

    void cmd_facts() {
        if (kb.facts.empty()) {
            std::cout << ansi::gray << "  No facts.\n" << ansi::reset;
            return;
        }
        std::cout << ansi::bold << "Facts (" << kb.facts.size() << "):\n" << ansi::reset;
        for (const auto& f : kb.facts) {
            std::cout << "  " << ansi::green << f.term.to_string() << ansi::reset
                      << ansi::gray << " [" << f.source << " | conf="
                      << std::fixed << std::setprecision(2) << f.confidence << "]\n" << ansi::reset;
        }
    }

    void cmd_rules() {
        if (kb.rules.empty()) {
            std::cout << ansi::gray << "  No rules.\n" << ansi::reset;
            return;
        }
        std::cout << ansi::bold << "Rules (" << kb.rules.size() << "):\n" << ansi::reset;
        for (const auto& r : kb.rules) {
            std::cout << "  " << ansi::cyan << r.name << ansi::reset
                      << " [fired=" << r.fire_count << "]\n";
            for (const auto& c : r.conditions)
                std::cout << ansi::gray << "    IF " << c.pattern.to_string() << "\n" << ansi::reset;
            std::cout << ansi::gray << "    THEN";
            for (const auto& p : r.conclusion_parts) std::cout << " " << p;
            std::cout << "\n" << ansi::reset;
        }
    }

    void cmd_derivations() {
        if (kb.derivations.empty()) {
            std::cout << ansi::gray << "  No derivations yet. Run 'think'.\n" << ansi::reset;
            return;
        }
        std::cout << ansi::bold << "Derivations (" << kb.derivations.size() << "):\n" << ansi::reset;
        for (const auto& d : kb.derivations) {
            std::cout << "  " << ansi::yellow << d.conclusion << ansi::reset
                      << ansi::gray << " [via " << d.rule_name << "]\n" << ansi::reset;
        }
    }

    void cmd_nn_train(const std::string& args) {
        auto tokens = split_tokens(args);
        if (tokens.size() < 2) {
            std::cout << ansi::red << "✗ Usage: nn_train <epochs> <lr>\n" << ansi::reset;
            return;
        }
        int epochs = std::stoi(tokens[0]);
        double lr = std::stod(tokens[1]);
        nn.lr = lr;

        std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
        for (const auto& f : kb.facts) {
            std::vector<double> feat(8, 0.0);
            for (size_t i = 0; i < f.term.functor.size() && i < 4; i++)
                feat[i] = (double)f.term.functor[i] / 128.0;
            for (size_t i = 0; i < f.term.args.size() && i < 2; i++)
                for (size_t j = 0; j < f.term.args[i].size() && j < 2; j++)
                    feat[4 + i*2 + j] = (double)f.term.args[i][j] / 128.0;
            double conf = f.confidence;
            dataset.push_back({feat, {conf, 1.0-conf, 0.5, 0.5}});
        }

        if (dataset.empty()) {
            std::cout << ansi::yellow << "⚠ No facts to train on.\n" << ansi::reset;
            return;
        }

        double total_loss = 0;
        for (int e = 0; e < epochs; e++) {
            std::shuffle(dataset.begin(), dataset.end(), rng);
            for (const auto& [x, y] : dataset)
                total_loss += nn.train(x, y);
        }
        std::cout << ansi::green << "✓ Trained " << epochs << " epochs | avg loss: "
                  << std::fixed << std::setprecision(6) << (total_loss / (epochs * dataset.size()))
                  << "\n" << ansi::reset;
    }

    void cmd_nn_info() {
        std::cout << ansi::bold << "Neural Network:\n" << ansi::reset;
        std::cout << ansi::gray << "  " << nn.info() << "\n" << ansi::reset;
        for (size_t i = 0; i < nn.layers.size(); i++) {
            const auto& l = nn.layers[i];
            std::cout << "  Layer " << i << ": " << l.in_size << " → " << l.out_size;
            switch(l.act) {
                case Activation::ReLU: std::cout << " [ReLU]"; break;
                case Activation::Sigmoid: std::cout << " [Sigmoid]"; break;
                case Activation::Tanh: std::cout << " [Tanh]"; break;
                case Activation::Linear: std::cout << " [Linear]"; break;
            }
            double wsum = 0;
            for (const auto& row : l.W) for (auto w : row) wsum += std::abs(w);
            std::cout << " | |W|=" << std::fixed << std::setprecision(3) << wsum << "\n";
        }
    }

    void cmd_nn_predict(const std::string& args) {
        auto tokens = split_tokens(args);
        if (tokens.size() < 8) {
            std::cout << ansi::red << "✗ Usage: nn_predict f0 f1 f2 f3 f4 f5 f6 f7\n" << ansi::reset;
            return;
        }
        std::vector<double> x(8);
        for (int i = 0; i < 8; i++) x[i] = std::stod(tokens[i]);
        auto out = nn.forward(x);
        std::cout << ansi::cyan << "Output: [";
        for (size_t i = 0; i < out.size(); i++) {
            if (i) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << out[i];
        }
        std::cout << "]\n" << ansi::reset;
    }

    void cmd_embed_sim(const std::string& args) {
        auto tokens = split_tokens(args);
        if (tokens.size() < 2) {
            std::cout << ansi::red << "✗ Usage: embed_sim word1 word2\n" << ansi::reset;
            return;
        }
        float sim = emb.similarity(tokens[0], tokens[1]);
        std::cout << ansi::cyan << "Similarity(" << tokens[0] << ", " << tokens[1] << ") = "
                  << std::fixed << std::setprecision(4) << sim << "\n" << ansi::reset;
    }

    void cmd_memory() {
        auto top = mem.top_k(10);
        std::cout << ansi::bold << "Memory (top " << top.size() << "):\n" << ansi::reset;
        for (const auto& e : top)
            std::cout << "  " << ansi::yellow << e.key << ansi::reset
                      << ansi::gray << " → " << e.value
                      << " [w=" << std::fixed << std::setprecision(2) << e.weight << "]\n" << ansi::reset;
    }

    void cmd_history() {
        std::cout << ansi::bold << "History (" << kb.history.size() << "):\n" << ansi::reset;
        int start = std::max(0, (int)kb.history.size() - 20);
        for (int i = start; i < (int)kb.history.size(); i++)
            std::cout << ansi::gray << "  " << (i+1) << ". " << kb.history[i] << "\n" << ansi::reset;
    }

    void cmd_forget(const std::string& args) {
        auto tokens = split_tokens(args);
        double threshold = tokens.empty() ? 0.3 : std::stod(tokens[0]);
        int before = kb.facts.size();
        kb.forget_weak(threshold);
        int after = kb.facts.size();
        std::cout << ansi::yellow << "⚡ Forgot " << (before - after) << " low-confidence facts (threshold=" << threshold << ")\n" << ansi::reset;
    }

    void cmd_concepts() {
        auto freq = kb.concept_frequency();
        std::vector<std::pair<int,std::string>> sorted;
        for (const auto& [k, v] : freq) sorted.push_back({v, k});
        std::sort(sorted.rbegin(), sorted.rend());
        std::cout << ansi::bold << "Concept frequency:\n" << ansi::reset;
        for (auto& [cnt, name] : sorted)
            std::cout << "  " << ansi::cyan << name << ansi::reset
                      << ansi::gray << " × " << cnt << "\n" << ansi::reset;
    }

    void cmd_save(const std::string& args) {
        std::string path = trim(args);
        if (path.empty()) path = "crappybat.model";
        nn.save(path);
        std::cout << ansi::green << "✓ Saved model to: " << path << "\n" << ansi::reset;
    }

    void cmd_load(const std::string& args) {
        std::string path = trim(args);
        if (path.empty()) path = "crappybat.model";
        if (nn.load(path)) {
            std::cout << ansi::green << "✓ Loaded model from: " << path << "\n" << ansi::reset;
        } else {
            std::cout << ansi::red << "✗ Could not load: " << path << "\n" << ansi::reset;
        }
    }

    void cmd_setname(const std::string& args) {
        session_name = trim(args);
        if (session_name.empty()) session_name = "Pipsqueak";
        std::cout << ansi::green << "✓ Session name: " << session_name << "\n" << ansi::reset;
    }

    void cmd_analogy() {
        auto analogies = kb.infer_analogies();
        if (analogies.empty()) {
            std::cout << ansi::gray << "  No analogies found.\n" << ansi::reset;
        } else {
            std::cout << ansi::bold << "Analogies:\n" << ansi::reset;
            for (const auto& a : analogies)
                std::cout << "  " << ansi::cyan << a << "\n" << ansi::reset;
        }
    }

    void cmd_help() {
        struct Cmd { std::string name, desc; };
        std::vector<Cmd> cmds = {
            {"add_fact <term>",            "Add a fact, e.g. sky(color, blue)"},
            {"add_rule \"name\" if ... then ...", "Add an inference rule"},
            {"think",                      "Run forward chaining inference"},
            {"query <statement>",          "Check if a statement is true"},
            {"explain <statement>",        "Show reasoning chain for a statement"},
            {"facts",                      "List all known facts"},
            {"rules",                      "List all rules"},
            {"derivations",                "List all derived conclusions"},
            {"analogy",                    "Show concept analogies"},
            {"concepts",                   "Show concept frequency map"},
            {"forget [threshold]",         "Forget low-confidence facts"},
            {"history",                    "Show session history"},
            {"memory",                     "Show memory store"},
            {"nn_info",                    "Show neural network architecture"},
            {"nn_train <epochs> <lr>",     "Train NN on known facts"},
            {"nn_predict f0..f7",          "Run NN forward pass"},
            {"embed_sim word1 word2",       "Cosine similarity of embeddings"},
            {"save [file]",                "Save NN model"},
            {"load [file]",                "Load NN model"},
            {"setname <name>",             "Set session name"},
            {"help",                       "Show this help"},
            {"exit / quit",                "Quit"},
        };

        std::cout << ansi::bold << "\nCommands:\n" << ansi::reset;
        for (const auto& c : cmds)
            std::cout << "  " << ansi::cyan << std::left << std::setw(40) << c.name
                      << ansi::reset << ansi::gray << c.desc << "\n" << ansi::reset;
        std::cout << "\n";
    }

    void run_command(const std::string& line) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') return;

        cmd_count++;
        std::string cmd, args;
        size_t sp = trimmed.find(' ');
        if (sp == std::string::npos) {
            cmd = trimmed;
        } else {
            cmd = trimmed.substr(0, sp);
            args = trim(trimmed.substr(sp + 1));
        }

        if      (cmd == "add_fact")    cmd_add_fact(args);
        else if (cmd == "add_rule")    cmd_add_rule(trimmed.substr(sp == std::string::npos ? trimmed.size() : sp + 1));
        else if (cmd == "think")       cmd_think();
        else if (cmd == "query")       cmd_query(args);
        else if (cmd == "explain")     cmd_explain(args);
        else if (cmd == "facts")       cmd_facts();
        else if (cmd == "rules")       cmd_rules();
        else if (cmd == "derivations") cmd_derivations();
        else if (cmd == "analogy")     cmd_analogy();
        else if (cmd == "concepts")    cmd_concepts();
        else if (cmd == "forget")      cmd_forget(args);
        else if (cmd == "history")     cmd_history();
        else if (cmd == "memory")      cmd_memory();
        else if (cmd == "nn_info")     cmd_nn_info();
        else if (cmd == "nn_train")    cmd_nn_train(args);
        else if (cmd == "nn_predict")  cmd_nn_predict(args);
        else if (cmd == "embed_sim")   cmd_embed_sim(args);
        else if (cmd == "save")        cmd_save(args);
        else if (cmd == "load")        cmd_load(args);
        else if (cmd == "setname")     cmd_setname(args);
        else if (cmd == "help")        cmd_help();
        else if (cmd == "exit" || cmd == "quit") {
            std::cout << ansi::gray << "Goodbye.\n" << ansi::reset;
            std::exit(0);
        }
        else {
            std::cout << ansi::red << "✗ Unknown command: " << cmd
                      << ". Type 'help' for commands.\n" << ansi::reset;
        }
    }

    void repl() {
        print_banner();
        std::string line;
        while (true) {
            print_prompt();
            if (!std::getline(std::cin, line)) {
                std::cout << "\n";
                break;
            }
            run_command(line);
        }
    }
};

int main(int argc, char* argv[]) {
    CrappyBat bat;

    if (argc > 1) {
        std::ifstream script(argv[1]);
        if (!script) {
            std::cerr << ansi::red << "✗ Cannot open script: " << argv[1] << "\n" << ansi::reset;
            return 1;
        }
        std::string line;
        while (std::getline(script, line)) {
            std::cout << ansi::gray << "» " << line << "\n" << ansi::reset;
            bat.run_command(line);
        }
        return 0;
    }

    bat.repl();
    return 0;
}
