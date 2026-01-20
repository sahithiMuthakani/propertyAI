/**
 * User Management System - Interactive JavaScript
 * ================================================
 * 
 * Features:
 * - Single Page Application (SPA) routing
 * - Form validation and submission
 * - API integration
 * - Dynamic UI updates
 * - Toast notifications
 * - Smooth animations
 * 
 * Author: AI Agent
 * Date: 2025-12-04
 */

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const AppState = {
    currentUser: null,
    currentView: 'login',
    users: [],
    stats: {}
};

// ============================================================================
// API SERVICE
// ============================================================================

const API = {
    baseURL: '',

    async request(endpoint, options = {}) {
        try {
            const response = await fetch(this.baseURL + endpoint, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Request failed');
            }

            return data;
        } catch (error) {
            throw error;
        }
    },

    async register(userData) {
        return this.request('/api/register', {
            method: 'POST',
            body: JSON.stringify(userData)
        });
    },

    async login(credentials) {
        return this.request('/api/login', {
            method: 'POST',
            body: JSON.stringify(credentials)
        });
    },

    async logout() {
        return this.request('/api/logout', {
            method: 'POST'
        });
    },

    async getCurrentUser() {
        return this.request('/api/me');
    },

    async getUsers() {
        return this.request('/api/users');
    },

    async getUser(userId) {
        return this.request(`/api/users/${userId}`);
    },

    async updateUser(userId, userData) {
        return this.request(`/api/users/${userId}`, {
            method: 'PUT',
            body: JSON.stringify(userData)
        });
    },

    async deleteUser(userId) {
        return this.request(`/api/users/${userId}`, {
            method: 'DELETE'
        });
    },

    async getStats() {
        return this.request('/api/stats');
    }
};

// ============================================================================
// UI UTILITIES
// ============================================================================

const UI = {
    showToast(message, duration = 3000) {
        const toast = document.getElementById('toast');
        const content = toast.querySelector('.toast-content');

        content.textContent = message;
        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
        }, duration);
    },

    showError(elementId, message) {
        const el = document.getElementById(elementId);
        if (el) {
            el.textContent = message;
            el.classList.add('show');
        }
    },

    hideError(elementId) {
        const el = document.getElementById(elementId);
        if (el) {
            el.classList.remove('show');
        }
    },

    showSuccess(elementId, message) {
        const el = document.getElementById(elementId);
        if (el) {
            el.textContent = message;
            el.classList.add('show');
        }
    },

    hideSuccess(elementId) {
        const el = document.getElementById(elementId);
        if (el) {
            el.classList.remove('show');
        }
    },

    setLoading(buttonId, loading) {
        const btn = document.getElementById(buttonId);
        if (btn) {
            if (loading) {
                btn.classList.add('loading');
                btn.disabled = true;
            } else {
                btn.classList.remove('loading');
                btn.disabled = false;
            }
        }
    },

    switchView(viewName) {
        // Hide all views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
        });

        // Show target view
        const targetView = document.getElementById(viewName + 'View');
        if (targetView) {
            targetView.classList.add('active');
            AppState.currentView = viewName;
        }

        // Show/hide header based on view
        const header = document.getElementById('header');
        if (viewName === 'login' || viewName === 'register') {
            header.style.display = 'none';
        } else {
            header.style.display = 'flex';
        }
    },

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
};

// ============================================================================
// AUTHENTICATION
// ============================================================================

const Auth = {
    async handleLogin(event) {
        event.preventDefault();

        UI.hideError('loginError');
        UI.setLoading('btnLogin', true);

        const email = document.getElementById('loginEmail').value.trim();
        const password = document.getElementById('loginPassword').value;

        try {
            const response = await API.login({ email, password });
            AppState.currentUser = response.user;

            UI.showToast('Login successful! Welcome back.');

            // Switch to dashboard
            await Dashboard.load();
            UI.switchView('dashboard');

        } catch (error) {
            UI.showError('loginError', error.message);
        } finally {
            UI.setLoading('btnLogin', false);
        }
    },

    async handleRegister(event) {
        event.preventDefault();

        UI.hideError('registerError');
        UI.setLoading('btnRegister', true);

        const username = document.getElementById('registerUsername').value.trim();
        const fullName = document.getElementById('registerFullName').value.trim();
        const email = document.getElementById('registerEmail').value.trim();
        const password = document.getElementById('registerPassword').value;

        // Validation
        if (password.length < 6) {
            UI.showError('registerError', 'Password must be at least 6 characters long');
            UI.setLoading('btnRegister', false);
            return;
        }

        try {
            const response = await API.register({
                username,
                full_name: fullName || null,
                email,
                password
            });

            AppState.currentUser = response.user;

            UI.showToast('Account created successfully! Welcome aboard.');

            // Switch to dashboard
            await Dashboard.load();
            UI.switchView('dashboard');

        } catch (error) {
            UI.showError('registerError', error.message);
        } finally {
            UI.setLoading('btnRegister', false);
        }
    },

    async handleLogout() {
        try {
            await API.logout();
            AppState.currentUser = null;
            AppState.users = [];
            AppState.stats = {};

            UI.showToast('Logged out successfully');
            UI.switchView('login');

            // Clear forms
            document.getElementById('loginForm').reset();

        } catch (error) {
            console.error('Logout error:', error);
            // Even if API fails, clear local state
            AppState.currentUser = null;
            UI.switchView('login');
        }
    }
};

// ============================================================================
// DASHBOARD
// ============================================================================

const Dashboard = {
    async load() {
        try {
            // Load stats
            const statsData = await API.getStats();
            AppState.stats = statsData;
            this.renderStats(statsData);

            // Load users
            const usersData = await API.getUsers();
            AppState.users = usersData.users;
            this.renderUsers(usersData.users);

        } catch (error) {
            UI.showToast('Error loading dashboard: ' + error.message);
        }
    },

    renderStats(stats) {
        document.getElementById('statTotalUsers').textContent = stats.total_users || 0;
        document.getElementById('statActiveUsers').textContent = stats.active_users || 0;
        document.getElementById('statInactiveUsers').textContent = stats.inactive_users || 0;
    },

    renderUsers(users) {
        const tbody = document.getElementById('usersTableBody');

        if (!users || users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="table-empty">No users found</td></tr>';
            return;
        }

        tbody.innerHTML = users.map(user => `
            <tr>
                <td>${user.id}</td>
                <td>${user.username}</td>
                <td>${user.email}</td>
                <td>${user.full_name || '-'}</td>
                <td><span class="badge badge-${user.role === 'admin' ? 'active' : 'inactive'}">${user.role}</span></td>
                <td><span class="badge badge-${user.is_active ? 'active' : 'inactive'}">${user.is_active ? 'Active' : 'Inactive'}</span></td>
                <td>${UI.formatDate(user.created_at)}</td>
                <td>
                    ${user.id !== AppState.currentUser.id ?
                `<button class="btn btn-sm btn-logout" onclick="Dashboard.handleDeleteUser(${user.id})">Delete</button>` :
                '<span style="color: var(--text-tertiary)">-</span>'
            }
                </td>
            </tr>
        `).join('');
    },

    async handleDeleteUser(userId) {
        if (!confirm('Are you sure you want to delete this user?')) {
            return;
        }

        try {
            await API.deleteUser(userId);
            UI.showToast('User deleted successfully');

            // Reload dashboard
            await this.load();

        } catch (error) {
            UI.showToast('Error deleting user: ' + error.message);
        }
    }
};

// ============================================================================
// PROFILE
// ============================================================================

const Profile = {
    async load() {
        try {
            // Get current user data
            const response = await API.getCurrentUser();
            AppState.currentUser = response.user;

            this.renderProfile(response.user);

        } catch (error) {
            UI.showToast('Error loading profile: ' + error.message);
        }
    },

    renderProfile(user) {
        // Fill form
        document.getElementById('profileUsername').value = user.username || '';
        document.getElementById('profileFullName').value = user.full_name || '';
        document.getElementById('profileEmail').value = user.email || '';
        document.getElementById('profilePassword').value = '';

        // Fill info
        document.getElementById('infoUserId').textContent = user.id;
        document.getElementById('infoUserRole').textContent = user.role;
        document.getElementById('infoUserCreated').textContent = UI.formatDate(user.created_at);
        document.getElementById('infoUserUpdated').textContent = UI.formatDate(user.updated_at);
    },

    async handleUpdate(event) {
        event.preventDefault();

        UI.hideError('profileError');
        UI.hideSuccess('profileSuccess');
        UI.setLoading('btnUpdateProfile', true);

        const username = document.getElementById('profileUsername').value.trim();
        const fullName = document.getElementById('profileFullName').value.trim();
        const email = document.getElementById('profileEmail').value.trim();
        const password = document.getElementById('profilePassword').value;

        const updateData = {
            username,
            full_name: fullName,
            email
        };

        if (password) {
            if (password.length < 6) {
                UI.showError('profileError', 'Password must be at least 6 characters long');
                UI.setLoading('btnUpdateProfile', false);
                return;
            }
            updateData.password = password;
        }

        try {
            await API.updateUser(AppState.currentUser.id, updateData);

            UI.showSuccess('profileSuccess', 'Profile updated successfully!');
            UI.showToast('Profile updated successfully');

            // Reload profile
            await this.load();

        } catch (error) {
            UI.showError('profileError', error.message);
        } finally {
            UI.setLoading('btnUpdateProfile', false);
        }
    }
};

// ============================================================================
// EVENT LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Auth forms
    document.getElementById('loginForm').addEventListener('submit', Auth.handleLogin);
    document.getElementById('registerForm').addEventListener('submit', Auth.handleRegister);

    // Auth links
    document.getElementById('linkToRegister').addEventListener('click', (e) => {
        e.preventDefault();
        UI.switchView('register');
    });

    document.getElementById('linkToLogin').addEventListener('click', (e) => {
        e.preventDefault();
        UI.switchView('login');
    });

    // Navigation
    document.getElementById('navDashboard').addEventListener('click', async () => {
        await Dashboard.load();
        UI.switchView('dashboard');
    });

    document.getElementById('navProfile').addEventListener('click', async () => {
        await Profile.load();
        UI.switchView('profile');
    });

    document.getElementById('btnLogout').addEventListener('click', Auth.handleLogout);

    // Dashboard
    document.getElementById('btnRefreshUsers').addEventListener('click', () => {
        Dashboard.load();
    });

    // Profile form
    document.getElementById('profileForm').addEventListener('submit', Profile.handleUpdate);

    // Initialize app
    UI.switchView('login');

    console.log('✓ User Management System Initialized');
    console.log('✓ Ready for user interaction');
});

// Make Dashboard available globally for inline event handlers
window.Dashboard = Dashboard;