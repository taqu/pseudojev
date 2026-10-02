#include "experiment/metrics.h"
#include <cstdio>
#include <cmath>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else { printf("pass: %s\n", what); }
}
static void check_near(double a, double b, double tol, const char* what) {
    check(std::fabs(a - b) < tol, what);
}

int main() {
    using namespace pjev;
    // -----------------------------------------------------------------------
    // Test 1: TypeMetrics accuracy
    // -----------------------------------------------------------------------
    {
        TypeMetrics tm;
        tm.labeled = 4;
        tm.correct = 3;
        check_near(tm.accuracy(), 0.75, 1e-9, "accuracy 3/4 = 0.75");
    }

    // -----------------------------------------------------------------------
    // Test 2: TypeMetrics accuracy — no labeled items
    // -----------------------------------------------------------------------
    {
        TypeMetrics tm;
        tm.labeled = 0;
        check_near(tm.accuracy(), -1.0, 1e-9, "accuracy with 0 labeled = -1");
    }

    // -----------------------------------------------------------------------
    // Test 3: TypeMetrics MAE
    // -----------------------------------------------------------------------
    {
        TypeMetrics tm;
        tm.labeled = 4;
        tm.abs_err = 2.0; // sum of |pred - actual| = 2.0
        check_near(tm.mae(), 0.5, 1e-9, "MAE = 2.0/4 = 0.5");
    }

    // -----------------------------------------------------------------------
    // Test 4: TypeMetrics QWK — perfect agreement
    // -----------------------------------------------------------------------
    {
        TypeMetrics tm;
        tm.labeled = 4;
        tm.predicted_levels = {0, 1, 2, 3};
        tm.actual_levels    = {0, 1, 2, 3};
        double k = tm.qwk(4);
        check_near(k, 1.0, 1e-6, "QWK perfect agreement = 1.0");
    }

    // -----------------------------------------------------------------------
    // Test 5: TypeMetrics QWK — anti-agreement (all wrong, systematic)
    // -----------------------------------------------------------------------
    {
        TypeMetrics tm;
        tm.labeled = 4;
        // Predicted opposite of actual for a 2-class scenario
        tm.predicted_levels = {0, 0, 1, 1};
        tm.actual_levels    = {1, 1, 0, 0};
        double k = tm.qwk(2);
        // With 2 classes, perfect anti-agreement => QWK = -1
        check(k < 0.0, "QWK anti-agreement < 0");
    }

    // -----------------------------------------------------------------------
    // Test 6: TypeMetrics QWK — too few labeled
    // -----------------------------------------------------------------------
    {
        TypeMetrics tm;
        tm.labeled = 1;
        double k = tm.qwk(4);
        check_near(k, -1.0, 1e-9, "QWK with labeled<2 = -1");
    }

    // -----------------------------------------------------------------------
    // Test 7: StabilityMetrics stability_rate
    // -----------------------------------------------------------------------
    {
        StabilityMetrics sm;
        sm.n_questions = 10;
        sm.n_stable    = 7;
        check_near(sm.stability_rate(), 0.7, 1e-9, "stability_rate 7/10 = 0.7");
    }

    // -----------------------------------------------------------------------
    // Test 8: StabilityMetrics stability_rate — no questions
    // -----------------------------------------------------------------------
    {
        StabilityMetrics sm;
        sm.n_questions = 0;
        check_near(sm.stability_rate(), -1.0, 1e-9, "stability_rate 0 questions = -1");
    }

    // -----------------------------------------------------------------------
    // Test 9: RunMetrics accumulation structure
    // -----------------------------------------------------------------------
    {
        RunMetrics m;
        m.n_total  = 10;
        m.n_errors = 2;
        m.choice.n       = 4;
        m.choice.labeled = 4;
        m.choice.correct = 3;
        m.noul.n       = 4;
        m.noul.labeled = 4;
        m.noul.correct = 4;
        m.score.n       = 2;
        m.score.labeled = 2;
        m.score.correct = 1;
        m.score.abs_err = 1.0;

        check_near(m.choice.accuracy(), 0.75, 1e-9, "RunMetrics choice accuracy");
        check_near(m.noul.accuracy(), 1.0, 1e-9, "RunMetrics noul accuracy");
        check_near(m.score.accuracy(), 0.5, 1e-9, "RunMetrics score accuracy");
        check_near(m.score.mae(), 0.5, 1e-9, "RunMetrics score MAE");
        check(m.n_total - m.n_errors == 8, "RunMetrics n_total - n_errors = 8");
    }

    // -----------------------------------------------------------------------
    // Test 10: TypeMetrics QWK — partial agreement, 3 levels
    // -----------------------------------------------------------------------
    {
        TypeMetrics tm;
        // 5 predictions: 3 perfect, 2 off-by-one
        tm.labeled = 5;
        tm.predicted_levels = {0, 1, 2, 1, 2};
        tm.actual_levels    = {0, 1, 2, 2, 1};
        double k = tm.qwk(3);
        // Off-by-one errors with 3 levels: w = 1/4 each, should be > 0
        check(k > 0.0 && k < 1.0, "QWK partial agreement in (0,1)");
    }

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
