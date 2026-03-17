let playersData = [];
let chartInstances = {};

// Chart.js Default styling for Dark Theme
Chart.defaults.color = '#8b949e';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(22, 27, 34, 0.9)';
Chart.defaults.plugins.tooltip.titleColor = '#e6edf3';
Chart.defaults.plugins.tooltip.bodyColor = '#e6edf3';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 8;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Fetch the generated data from Python scripts
        const [profRes, dlRes] = await Promise.all([
            fetch('/dashboard_data/top50_enriched_profiles.json'),
            fetch('/dashboard_data/deep_learning_metrics.json')
        ]);
        
        if (!profRes.ok) throw new Error('Failed to load player data');
        
        playersData = await profRes.json();
        
        if (dlRes.ok) {
            const dlData = await dlRes.json();
            document.getElementById('dlLstm').textContent = (dlData.lstm_momentum_accuracy * 100).toFixed(1) + '%';
            document.getElementById('dlTransformer').textContent = (dlData.transformer_attention_accuracy * 100).toFixed(1) + '%';
        }
        
        // Populate Sidebar
        renderPlayerList(playersData);
        
        // Search functionality
        document.getElementById('playerSearch').addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            const filtered = playersData.filter(p => p.player_name.toLowerCase().includes(term));
            renderPlayerList(filtered);
        });

        // Load first player by default
        if (playersData.length > 0) {
            selectPlayer(playersData[0]);
        }

    } catch (err) {
        console.error("Initialization Error:", err);
        document.querySelector('.main-content').innerHTML = `
            <div style="padding:2rem;text-align:center;">
                <h2>Error Loading Data</h2>
                <p>Could not load /dashboard_data/top50_enriched_profiles.json.</p>
                <p>Ensure you are running a local server at the Tennis_data root.</p>
                <code>python -m http.server 8000</code>
            </div>`;
    }
});

function renderPlayerList(players) {
    const listEl = document.getElementById('playerList');
    listEl.innerHTML = '';
    
    players.forEach((p, idx) => {
        const li = document.createElement('li');
        li.className = 'player-item';
        // Select logic
        li.onclick = () => {
            document.querySelectorAll('.player-item').forEach(el => el.classList.remove('active'));
            li.classList.add('active');
            selectPlayer(p);
        };
        
        // First item active by default on load
        if(idx === 0) li.classList.add('active');

        li.innerHTML = `
            <span>${p.player_name} <small style="color:var(--text-muted)">(${p.ioc})</small></span>
            <span style="color:var(--accent-blue)">#${idx+1}</span>
        `;
        listEl.appendChild(li);
    });
}

function selectPlayer(p) {
    // Header
    document.getElementById('playerName').textContent = p.player_name;
    document.getElementById('playerArchetype').textContent = p.archetype;
    document.getElementById('playerArchetype').style.display = 'inline-block';
    
    document.getElementById('winPct').textContent = (p.career_stats.win_pct * 100).toFixed(1) + '%';
    document.getElementById('dominanceRatio').textContent = p.overall_profile.dominance_ratio.toFixed(2);
    
    // Strengths & Weaknesses
    const ulStr = document.getElementById('strengthsList');
    ulStr.innerHTML = p.strengths.slice(0, 3).map(s => 
        `<li><span>${s.metric}</span> <span class="diff">${s.diff}</span></li>`
    ).join('');
    
    const ulWks = document.getElementById('weaknessesList');
    ulWks.innerHTML = p.weaknesses.slice(0, 3).map(w => 
        `<li><span>${w.metric}</span> <span class="diff">${w.diff}</span></li>`
    ).join('');

    // Update Charts
    updateRadarChart(p);
    updateSurfaceChart(p);
    updateTactics(p);
    updateCombos(p);
}

function updateRadarChart(p) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    if (chartInstances.radar) chartInstances.radar.destroy();
    
    // Normalize data roughly against typical max values to render a nice radar
    const s = p.serve_profile;
    const o = p.overall_profile;
    
    // Normalize percentages to 0-100 scale for plotting
    const data = [
        s.first_serve_pct * 100,           // Consistency
        s.first_serve_win_pct * 100,       // 1st Srv Power
        s.second_serve_win_pct * 100,      // 2nd Srv Safety
        o.return_pts_won_pct * 100 * 1.5,  // Return Game (Scaled up for viz)
        (1 - s.bp_save_pct) * 100,         // Clutch (Inverse representation visually adjusted)
        s.ace_rate * 100 * 5               // Ace Rate (Scaled for viz)
    ];

    chartInstances.radar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['1st Serve %', '1st Srv Win', '2nd Srv Win', 'Return Pts', 'Clutch Pt', 'Ace Power'],
            datasets: [{
                label: p.player_name,
                data: data,
                backgroundColor: 'rgba(88, 166, 255, 0.2)',
                borderColor: '#58a6ff',
                pointBackgroundColor: '#58a6ff',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: {
                        color: '#e6edf3',
                        font: { size: 11, family: 'Inter' }
                    },
                    ticks: { display: false, max: 100, min: 0 }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function updateSurfaceChart(p) {
    const ctx = document.getElementById('surfaceChart').getContext('2d');
    if (chartInstances.surface) chartInstances.surface.destroy();
    
    const splits = p.surface_splits;
    const surfaces = Object.keys(splits).map(s => s.charAt(0).toUpperCase() + s.slice(1));
    const winPcts = Object.values(splits).map(v => v.win_pct * 100);

    chartInstances.surface = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: surfaces,
            datasets: [{
                label: 'Win %',
                data: winPcts,
                backgroundColor: [
                    'rgba(88, 166, 255, 0.8)', // Hard - Blue
                    'rgba(248, 81, 73, 0.8)',  // Clay - Red
                    'rgba(63, 185, 80, 0.8)'   // Grass - Green
                ],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { callback: v => v + '%' }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function updateTactics(p) {
    const mcp = p.mcp_tactical_profile;
    
    if (!mcp || !mcp.mcp_data_exists) {
        document.getElementById('deuceWinPct').textContent = 'N/A';
        document.getElementById('serveDeuceWide').textContent = '-';
        document.getElementById('serveDeuceT').textContent = '-';
        if (chartInstances.rally) chartInstances.rally.destroy();
        return;
    }

    // Clutch Update
    const deucePct = (mcp.deuce_win_pct * 100).toFixed(1);
    const deuceEl = document.getElementById('deuceWinPct');
    deuceEl.textContent = deucePct + '%';
    deuceEl.style.color = deucePct > 63.3 ? 'var(--accent-green)' : 'var(--accent-red)';

    // Serve Split Update
    document.getElementById('serveDeuceWide').textContent = (mcp.serve_directions.deuce_wide_pct * 100).toFixed(0) + '%';
    document.getElementById('serveDeuceT').textContent = (mcp.serve_directions.deuce_t_pct * 100).toFixed(0) + '%';

    // Rally Chart
    const ctx = document.getElementById('rallyChart').getContext('2d');
    if (chartInstances.rally) chartInstances.rally.destroy();

    chartInstances.rally = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['1-3', '4-6', '7-9', '10+'],
            datasets: [{
                label: 'Win % by Rally Len',
                data: [
                    mcp.rally_win_pcts['1-3']*100,
                    mcp.rally_win_pcts['4-6']*100,
                    mcp.rally_win_pcts['7-9']*100,
                    mcp.rally_win_pcts['10']*100
                ],
                borderColor: '#bc8cff',
                backgroundColor: 'rgba(188, 140, 255, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 40, max: 70, display: false },
                x: { grid: { display: false }, ticks: { font: { size: 10 } } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function updateCombos(p) {
    const shotsList = document.getElementById('topShotsList');
    const combosList = document.getElementById('topCombosList');
    
    shotsList.innerHTML = '';
    if (p.top_shots && p.top_shots.length > 0) {
        p.top_shots.forEach(s => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="combo-name">${s.shot_name}</span>
                <div class="combo-stat">
                    <span class="combo-pct">${(s.win_pct * 100).toFixed(1)}%</span>
                    <span class="combo-sub">${s.shots} shots</span>
                </div>
            `;
            shotsList.appendChild(li);
        });
    } else {
        shotsList.innerHTML = '<li><span style="color:var(--text-muted)">No shot data available</span></li>';
    }

    combosList.innerHTML = '';
    if (p.top_combos && p.top_combos.length > 0) {
        p.top_combos.forEach(c => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="combo-name">${c.serve_plus_1}</span>
                <div class="combo-stat">
                    <span class="combo-pct">${(c.win_pct * 100).toFixed(1)}%</span>
                    <span class="combo-sub">${c.pts_played} points</span>
                </div>
            `;
            combosList.appendChild(li);
        });
    } else {
        combosList.innerHTML = '<li><span style="color:var(--text-muted)">No combo data available</span></li>';
    }
}
