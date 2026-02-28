import { useState, useEffect } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react'

import { navLinks } from '../../constants/index.js'

const Navbar = () => {
	const [theme, setTheme] = useState(localStorage.getItem('blucare_theme') || 'dark');
	const [isLoggedIn, setIsLoggedIn] = useState(localStorage.getItem('blucare_logged_in') === 'true' || true);
	const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

	useEffect(() => {
		document.documentElement.setAttribute('data-theme', theme);
		localStorage.setItem('blucare_theme', theme);
	}, [theme]);

	const toggleTheme = () => {
		setTheme(prev => prev === 'dark' ? 'light' : 'dark');
	};

	const toggleMobileMenu = () => {
		setIsMobileMenuOpen(!isMobileMenuOpen);
	};

	useGSAP(() => {
		if (isMobileMenuOpen) {
			gsap.to('.mobile-shelf', { x: 0, duration: 0.5, ease: 'power3.out' });
		} else {
			gsap.to('.mobile-shelf', { x: '100%', duration: 0.5, ease: 'power3.in' });
		}
	}, [isMobileMenuOpen]);

	useGSAP(() => {
		const navTween = gsap.timeline({
			scrollTrigger: {
				trigger: 'nav',
				start: 'bottom top'
			}
		});

		navTween.fromTo('nav',
			{ backgroundColor: 'transparent', backdropFilter: 'blur(0px)' },
			{
				backgroundColor: 'var(--nav-bg)',
				backdropFilter: 'blur(10px)',
				duration: 0.6,
				ease: 'power2.inOut'
			}
		);
	})

	return (
		<>
			<nav className="fixed z-50 w-full transition-all duration-300 h-20 px-5 md:px-10 flex items-center justify-between">
				<div className="flex items-center justify-between w-full">
					<a href="#home" className="flex items-center gap-3 no-underline text-primary font-medium text-[1.1rem] tracking-[-0.01em]">
						<div className="brand-dot flex-shrink-0"></div>
						BluCare+
					</a>

					<div className="flex items-center gap-4 lg:gap-10">
						<div className="nav-links flex items-center gap-8">
							<a
								href="chat.html"
								className="cursor-pointer text-nowrap text-[0.9rem] transition-colors text-subdued hover:text-lavender no-underline"
							>
								Chat
							</a>
							<a
								href="hospitals.html"
								className="cursor-pointer text-nowrap text-[0.9rem] transition-colors text-subdued hover:text-lavender no-underline"
							>
								For Hospitals
							</a>

							<button
								onClick={toggleTheme}
								className="theme-toggle"
								title="Toggle Theme"
							>
								<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
									{theme === 'light' ? (
										<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
									) : (
										<>
											<circle cx="12" cy="12" r="5"></circle>
											<line x1="12" y1="1" x2="12" y2="3"></line>
											<line x1="12" y1="21" x2="12" y2="23"></line>
											<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
											<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
											<line x1="1" y1="12" x2="3" y2="12"></line>
											<line x1="21" y1="12" x2="23" y2="12"></line>
											<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
											<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
										</>
									)}
								</svg>
							</button>
						</div>

						<div className="user-profile hidden md:flex">
							{isLoggedIn ? (
								<>
									<div className="user-avatar text-[0.85rem] font-medium">P</div>
									<div className="user-name text-[0.85rem] font-normal">Hi, Prashik</div>
								</>
							) : (
								<div className="user-avatar neutral">
									<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
										<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
										<circle cx="12" cy="7" r="4"></circle>
									</svg>
								</div>
							)}
						</div>

						<button onClick={toggleMobileMenu} className="mobile-menu-btn" aria-label="Toggle Menu">
							<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
								{isMobileMenuOpen ? (
									<path d="M18 6L6 18M6 6l12 12" />
								) : (
									<path d="M4 6h16M4 12h16M4 18h16" />
								)}
							</svg>
						</button>
					</div>
				</div>
			</nav>

			{/* Mobile Menu Shelf */}
			<div className={`mobile-shelf fixed inset-y-0 right-0 w-64 bg-bg-surface/95 backdrop-blur-xl z-[60] transform translate-x-full border-l border-light md:hidden`}>
				<div className="flex flex-col h-full p-8 pt-24 space-y-8">
					<a href="chat.html" className="text-xl font-medium text-primary no-underline" onClick={toggleMobileMenu}>Chat</a>
					<a href="hospitals.html" className="text-xl font-medium text-primary no-underline" onClick={toggleMobileMenu}>For Hospitals</a>
					<div className="pt-8 border-t border-light mt-auto">
						{isLoggedIn ? (
							<div className="flex items-center gap-3">
								<div className="user-avatar">P</div>
								<div className="user-name">Hi, Prashik</div>
							</div>
						) : (
							<div className="user-avatar neutral">
								<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
									<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
									<circle cx="12" cy="7" r="4"></circle>
								</svg>
							</div>
						)}

						<button onClick={toggleTheme} className="flex items-center gap-3 mt-8 text-secondary">
							<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
								{theme === 'light' ? <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path> : <circle cx="12" cy="12" r="5"></circle>}
							</svg>
							{theme === 'light' ? 'Dark Mode' : 'Light Mode'}
						</button>
					</div>
				</div>
			</div>
			{isMobileMenuOpen && (
				<div className="fixed inset-0 bg-black/50 z-[55] md:hidden" onClick={toggleMobileMenu} />
			)}
		</>
	)
}
export default Navbar
